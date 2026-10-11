"""See node."""
import random
from datetime import datetime

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