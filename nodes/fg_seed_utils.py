"""See node."""
import torch
import random
from datetime import datetime

import node_helpers
import comfy.sample as c_sample

from ._fg_helperfunctions import log, get_worflow_node

# Some extension must be setting a seed as server-generated seeds were not random. We'll set a new
# seed and use that state going forward.
initial_random_state = random.getstate()
random.seed(datetime.now().timestamp())
rgthree_seed_random_state = random.getstate()
random.setstate(initial_random_state)


def new_random_seed():
	""" Gets a new random seed from the rgthree_seed_random_state and resetting the previous state."""
	global rgthree_seed_random_state
	prev_random_state = random.getstate()
	random.setstate(rgthree_seed_random_state)
	seed = random.randint(1, 1125899906842624)
	rgthree_seed_random_state = random.getstate()
	random.setstate(prev_random_state)
	return seed


class FG_Noise:
	"""The object that travels down a NOISE link.

	A NOISE output is *not* a tensor. Comfy has no idea what shape the noise should be at the
	time this node runs, so the link carries a small lazy generator instead. The sampler
	(SamplerCustom / SamplerCustomAdvanced / anything taking a NOISE input) calls
	`generate_noise(latent)` itself, at sample time, with the LATENT it was given. That is why
	the stock RandomNoise node has no latent input.

	The contract is just two things:
	  - a `.seed` attribute (samplers read it for previews / metadata)
	  - a `.generate_noise(input_latent)` method returning a CPU tensor shaped like
	    input_latent["samples"]

	`input_latent` is the raw LATENT dict, i.e. {"samples": Tensor, "batch_index": [ints]?}.
	"""

	def __init__(self, seed):
		self.seed = seed

	def generate_noise(self, input_latent):
		"""Builds the actual noise tensor once the sampler knows the latent shape."""
		latent_image = input_latent["samples"]
		batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
		return c_sample.prepare_noise(latent_image, self.seed, batch_inds)


class FG_NoisySeeder:
	"""Seed node."""
	def __init__(self):
		self.NODE_NAME = "Noisy Seed"

	@classmethod
	def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
		return {
		"required": {
			"seed": ("INT", {"default": 0, "min": -1125899906842624, "max": 1125899906842624}),
		},
		"hidden": {
				"prompt": "PROMPT",
				"extra_pnginfo": "EXTRA_PNGINFO",
				"unique_id": "UNIQUE_ID",
			},
		}

	RETURN_TYPES = ("INT", "NOISE", )
	RETURN_NAMES = ("Seed", "Noise", )
	OUTPUT_TOOLTIPS = (
		"🎲 The seed that was actually used for this run.",
		"✨ Noise generator built from that same seed. Feed it to SamplerCustom / "
		"SamplerCustomAdvanced or any node taking a NOISE input.",
	)
	FUNCTION = "main"
	CATEGORY = "Farrenzo's Garbage/Sampling"

	@classmethod
	def IS_CHANGED(cls, seed, prompt=None, extra_pnginfo=None, unique_id=None):
		"""Forces a changed state if we happen to get a special seed, as if from the API directly."""
		if seed in (-1, -2, -3):
		# This isn't used, but a different value than previous will force it to be "changed"
			return new_random_seed()
		return seed

	def main(self, seed=0, prompt=None, extra_pnginfo=None, unique_id=None):
		"""Returns the passed seed on execution."""

		# We generate random seeds on the frontend in the seed node before sending the workflow in for
		# many reasons. However, if we want to use this in an API call without changing the seed before
		# sending, then users _could_ pass in "-1" and get a random seed used and added to the metadata.
		# Though, this should likely be discouraged for several reasons (thus, a lot of logging).
		if seed in (-1, -2, -3):
			log(
				f"{self.NODE_NAME}: Got '{seed}' as passed seed. This shouldn't happen when queueing from the ComfyUI frontend.",
				message_type="warning"
			)
		if seed in (-2, -3):
			log(
				f"{self.NODE_NAME}: Got '{seed}' as passed seed. Cannot {'increment' if seed == -2 else 'decrement'} seed from server, but will generate a new random seed.",
    			message_type="warning"
			)
		original_seed = seed
		seed = new_random_seed()
		log(
			f"{self.NODE_NAME}: Server-generated random seed '{seed}' and saving to workflow.",
			message_type="info"
		)
		log(
			f"{self.NODE_NAME} NOTE: Re-queues passing in '{seed}' and server-generated random seed won't be cached.",
			message_type="warning"
		)

		if unique_id is None:
			log(
			    f"{self.NODE_NAME} Cannot save server-generated seed to image metadata because the node\'s id was not provided.",
			    message_type="warning"
			)
		else:
			if extra_pnginfo is None:
				log(
					f"{self.NODE_NAME} Cannot save server-generated seed to image workflow metadata because workflow was not provided.",
					message_type="warning"
				)
			else:
				log(f"{self.NODE_NAME} Looking for Seed node with id '{unique_id}'", message_type="info")
				workflow_node = get_worflow_node(extra_pnginfo, str(unique_id))
				if workflow_node is None or 'widgets_values' not in workflow_node:
					log(
						f"{self.NODE_NAME} Cannot save server-generated seed to image workflow metadata because node was not found in the provided workflow.",
					    message_type="warning"
					)
				else:
					for index, widget_value in enumerate(workflow_node['widgets_values']):
						if widget_value == original_seed:
							workflow_node['widgets_values'][index] = seed

			if prompt is None:
				log(
					f"{self.NODE_NAME} Cannot save server-generated seed to image API prompt metadata because prompt was not provided.",
					message_type="warning"
				)
			else:
				prompt_node = prompt.get(str(unique_id))
				if prompt_node is None or 'inputs' not in prompt_node or 'seed' not in prompt_node['inputs']:
					log(
						f"{self.NODE_NAME} Cannot save server-generated seed to image workflow metadata because node was not found in the provided workflow.",
					    message_type="warning"
					)
				else:
					prompt_node['inputs']['seed'] = seed

		return (seed, FG_Noise(seed), )


class FG_SeedVarianceEnhancer:
    def __init__(self):
        self.NODE_NAME = "FG Seed Variance Enhancer"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "randomize_percent": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 100.0, "step": 1, "tooltip": "The percentage of embedding values to which random noise is added."}),
                "strength": ("FLOAT", {"default": 20, "min": -0xFFFFFFFF, "max": 0xFFFFFFFF, "step": 0.00001, "tooltip": "The scale of the random noise."}),
                "noise_insert": (["noise on beginning steps", "noise on ending steps", "noise on all steps", "disabled"], {"tooltip": "On which steps of the generation process the noisy text embedding is used."}),
                "steps_switchover_percent": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 99.0, "step": 1, "tooltip": "The percentage of steps processed before the switch between noisy and original embeddings occurs. Use this formula: (100/TOTALSTEPS) * STEPS - 1"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True, "tooltip": "The random seed used for embedding value selection and noise generation. Add 1 billion to strength to revert to v2.1 seed behavior."}),
                "mask_starts_at": (["beginning", "end"], {"tooltip": "Which end of the prompt will be protected from noise. The mask that blocks noise will expand from the specified end a percentage of the way towards the opposite end."}),
                "mask_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 99.0, "step": 1, "tooltip": "The percentage of the prompt that will be protected from noise. Settings of \"beginning\" and \"50\" results in the first half of the prompt being protected."}),
                "log_to_console": ("BOOLEAN", {"default": False, "tooltip": f"Print out useful information to the ComfyUI console, including suggested strength values."})
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "randomize_conditioning"
    CATEGORY = "Farrenzo's Garbage/Sampling"

    #prints to console statistics about a tensor
    def log_tensor_statistics(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            log(f"{self.NODE_NAME}received a conditioning with no Tensor.", message_type="warning")
            return

        # Find null sequences
        first_null, last_nonnull, null_sequences = self.tensor_first_null_sequence(tensor)

        if last_nonnull < tensor.size(1) - 1:
            # Slice tensor up to the last_nonnull layer (inclusive) along sequence dimension
            sliced_tensor = tensor[:, :last_nonnull + 1, :]
            # Calculate statistics
            mean = torch.mean(sliced_tensor).item()
            std = torch.std(sliced_tensor).item()
            min_val = torch.min(sliced_tensor).item()
            max_val = torch.max(sliced_tensor).item()
        else:
            # Calculate statistics
            mean = torch.mean(tensor).item()
            std = torch.std(tensor).item()
            min_val = torch.min(tensor).item()
            max_val = torch.max(tensor).item()

        # Log statistics to console
        log(f"{self.NODE_NAME}: Embedding Tensor Statistics:  (from SeedVarianceEnhancer)")
        if first_null != -1:
            # count null sequences
            number_of_null_seq = 0
            for i in null_sequences:
                if i == 0:
                    number_of_null_seq += 1
            log(f"{self.NODE_NAME}: null sequences:   index of first: {first_null}   Index of last nonnull: {last_nonnull}   total: {number_of_null_seq}.")
        log(f"{self.NODE_NAME}: Dimensions: {', '.join(map(str, tensor.shape))}   Min: {min_val:.6f}   Max: {max_val:.6f}   Mean: {mean:.6f}   Standard Deviation: {std:.6f}   Try strength in range {std/10:.6f} - {std*10:.6f}")


    # searches for sequences within a tensor that contain all null bytes
    def tensor_first_null_sequence(self, tensor):
        first_null = -1  # Index of the first sequence that is all zeros
        last_nonnull = -1 # Index of the last sequence that has nonzero values
        null_sequences = [0] * tensor.size(1)  # Initialize array with 0s

        if tensor.dim() == 3:  # Ensure tensor has 3 dimensions
            for i in range(tensor.size(1)):  # Iterate through each sequence in the second dimension
                sequence = tensor[:, i, ...]  # Extract current sequence
                is_all_zero = torch.all(sequence == 0)  # Check if the sequence is all zeros

                # Update the null_sequences array
                null_sequences[i] = 0 if is_all_zero else 1

                if not is_all_zero:
                    last_nonnull = i

                # Track the first null sequence
                if is_all_zero and first_null == -1:
                    first_null = i

        return (first_null, last_nonnull, null_sequences)


    def randomize_conditioning(self, conditioning, randomize_percent, strength, noise_insert, steps_switchover_percent, seed, mask_starts_at, mask_percent, log_to_console):
        # Validate and scale input
        steps_switchover_percent = max(1, min(99, steps_switchover_percent)) / 100
        randomize_percent = max(1, min(100, randomize_percent)) / 100
        mask_percent = max(0, min(99, mask_percent)) / 100

        # Check for early return conditions
        if len(conditioning) < 1 or len(conditioning[0]) < 2 or ( len(conditioning) == 2 and len(conditioning[1]) < 2 ):
            if log_to_console:
                log(f"{self.NODE_NAME}: received an empty conditioning. Passing it through unchanged.", message_type="warning")
            return (conditioning,)
        if strength == 0:
            if log_to_console:
                log(f"{self.NODE_NAME} is disabled. Strength is set to zero. Passing conditioning through unchanged.", message_type="warning")
                self.log_tensor_statistics(conditioning[0][0])
            return (conditioning,)
        if noise_insert == "disabled":
            if log_to_console:
                log(f"{self.NODE_NAME}: is disabled. Passing conditioning through unchanged.", message_type="warning")
                self.log_tensor_statistics(conditioning[0][0])
            return (conditioning,)

        if len(conditioning) > 2 and log_to_console:
            log(f"{self.NODE_NAME}: will only use the first two embeddings from this conditioning.", message_type="warning")

        reset_seed = True # By default, we will use new seed behavior in version 2.2 and up
        if int(strength / 1000000000) == 1: # if the user added 1 billion to strength, we will revert to old seed behavior
            strength -= 1000000000
            reset_seed = False
            if log_to_console:
                log(f"{self.NODE_NAME} has detected 1 billion added to strength. Subtracting the 1B and reverting to v2.1 seed behavior.")

        # Select the embedding we will work with
        if len(conditioning) == 1:
            t = [ conditioning[0][0], conditioning[0][1].copy() ]
            t_other = [ conditioning[0][0], conditioning[0][1].copy() ]
        elif len(conditioning) >= 2:
            if noise_insert == "noise on beginning steps":
                t = [ conditioning[0][0], conditioning[0][1].copy() ]
                t_other = [ conditioning[1][0], conditioning[1][1].copy() ]
            elif noise_insert == "noise on ending steps":
                t = [ conditioning[1][0], conditioning[1][1].copy() ]
                t_other = [ conditioning[0][0], conditioning[0][1].copy() ]
            else: # Doing "noise on all steps." If upstream SVH nodes only added noise to the end, we will use that embedding
                if conditioning[0][1]["SVH_tag"] != "noisy" and conditioning[1][1]["SVH_tag"] == "noisy":
                    t = [ conditioning[1][0], conditioning[1][1].copy() ]
                    t_other = [ conditioning[1][0], conditioning[1][1].copy() ]
                else:
                    t = [ conditioning[0][0], conditioning[0][1].copy() ]
                    t_other = [ conditioning[0][0], conditioning[0][1].copy() ]


        if isinstance(t[0], torch.Tensor):
            if log_to_console:
                self.log_tensor_statistics(t[0]) # print statistical analysis of tensor to console

            torch.manual_seed(seed)
            noise = torch.rand_like(t[0]) * 2 * strength - strength
            if reset_seed:
                torch.manual_seed(seed+1)
            noise_mask = torch.bernoulli(torch.ones_like(t[0]) * randomize_percent).bool() # Randomly select a percentage of values.

            #check for null sequences
            first_null, last_nonnull, null_sequences = self.tensor_first_null_sequence(t[0])

            # Check if we need to apply masking logic based on mask_percent or null sequences
            if mask_percent > 0 or last_nonnull < t[0].size(1) - 1:
                if last_nonnull < t[0].size(1) - 1 and last_nonnull >= 0:
                    seq_len = last_nonnull + 1
                else:
                    seq_len = t[0].size(1)

                if mask_starts_at == "end":
                    mask_start = seq_len - int(seq_len * mask_percent)
                    mask_end = t[0].size(1)
                else:
                    mask_start = 0
                    mask_end = int(seq_len * mask_percent)

                # Create the mask
                prompt_mask = torch.arange(t[0].size(1), device=t[0].device).view(1, -1, 1).expand(t[0].size(0), -1, t[0].size(2))
                prompt_mask = (prompt_mask >= mask_start) & (prompt_mask < mask_end)

                if first_null > -1:  # There are some null sequences
                    if log_to_console:
                        log(f"{self.NODE_NAME}: is masking null sequence from noise.")

                    # Create null_mask from null_sequences, reshape, and expand
                    null_mask_tensor = ~torch.tensor(null_sequences, device=t[0].device, dtype=torch.bool)
                    null_mask_tensor = null_mask_tensor.view(1, -1, 1).expand(t[0].size(0), -1, t[0].size(2))

                    # Combine with existing mask to include null sequences in the protected region
                    prompt_mask = prompt_mask | null_mask_tensor  # Logical OR: protect both range and nulls

                # Combine with existing mask
                noise_mask = noise_mask & (~prompt_mask)  # Zeros noise_mask within the mask range and nulls

            modified_noise = noise * noise_mask # Only apply noise to the selected values.
            noisy_tensor = t[0] + modified_noise
            noisy_embedding = [ [noisy_tensor, t[1]] ]
            other_embedding = [ t_other ]

            if noise_insert == "noise on beginning steps":
                new_conditioning = node_helpers.conditioning_set_values(noisy_embedding, {"start_percent": 0.0, "end_percent": steps_switchover_percent, "SVH_tag" : "noisy"})
                new_conditioning += node_helpers.conditioning_set_values(other_embedding, {"start_percent": steps_switchover_percent, "end_percent": 1.0})
            elif noise_insert == "noise on ending steps":
                new_conditioning = node_helpers.conditioning_set_values(other_embedding, {"start_percent": 0.0, "end_percent": steps_switchover_percent})
                new_conditioning += node_helpers.conditioning_set_values(noisy_embedding, {"start_percent": steps_switchover_percent, "end_percent": 1.0, "SVH_tag" : "noisy"})
            else:
                if "start_percent" in noisy_embedding[0][1]:
                    del noisy_embedding[0][1]["start_percent"]
                if "end_percent" in noisy_embedding[0][1]:
                    del noisy_embedding[0][1]["end_percent"]
                return (noisy_embedding,)

            return (new_conditioning,)

        else: # if t[0] was not a Tensor
            if log_to_console:
                log(f"{self.NODE_NAME}: received a conditioning with no Tensor. Passing it through untouched.", message_type="warning")
            return (conditioning,)


SEED_UTILS_NODE_CLASS_MAPPINGS = {
    "FG_NoisySeeder"         : FG_NoisySeeder,
    "FG_SeedVarianceEnhancer": FG_SeedVarianceEnhancer,
}

SEED_UTILS_DISPLAY_NAME_MAPPINGS = {
    "FG_NoisySeeder"         : "🎲 Noisy Seeder",
    "FG_SeedVarianceEnhancer": "🎲 Seed Variance Enhancer",
}
