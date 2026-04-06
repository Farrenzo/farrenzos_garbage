"""
I have absolutely no idea what this does. Here's whgat CLAUDE said:

This node controls how reference images are injected into the
FLUX Kontext diffusion process. Remember in your CLIP encode node,
when you pass images + VAE, the VAE-encoded images get attached to
the conditioning as reference_latents. This node tells the model
how to use them during generation.

The methods:
1. Offset — Reference latents are concatenated to the end of the
main latent sequence with a positional offset. The model sees
them as extra context tokens "beside" the image being generated.
This is the most common approach — it's like pinning reference
photos next to your canvas.

2. Index — Reference latents are inserted at specific indices within
the latent sequence. More precise placement, giving the model explicit
positional information about where each reference belongs.

3. Index Timestep Zero — Same as index, but the reference latents are
marked as timestep zero, meaning "these are fully clean, already denoised
images." This tells the model to treat them as ground truth rather than
noisy inputs that need denoising. Useful for strict style/content transfer.

4. UXO/UNO — A specific injection method (the node normalizes both "uxo"
and "uso" strings to "uxo") that handles reference blending differently.
Used for UNO-style multi-reference generation.

Applying reference methods to negative conditioning is mostly meaningless
for FLUX since FLUX doesn't use traditional CFG. The negative path will
typically be zeroed-out conditioning with no reference latents attached anyway.

Other models like QWEN Image Edit, for example, would benefit from the negative
conditioning.
"""

global_description = """
This node controls how reference images are injected into the
FLUX Kontext diffusion process. This is best used if reference
images are passed with a VAE, that way, the VAE-encoded images
get attached to the conditioning as reference_latents. This node
tells the model how to use them during generation.
"""

class FG_ModelReferenceLatentMethod:
    def __init__(self):
        self.NODE_NAME = "Edit Model Reference Method"

    @classmethod
    def INPUT_TYPES(cls):
        reference_latent_methods = ["skip", "offset", "index", "uxo", "index_timestep_zero"]
        return {
            "required": {
                "positive_conditioning":            ("CONDITIONING", {"tooltip": "➕ Positive prompt conditioning info."}),
                "positive_reference_latent_method": (reference_latent_methods, {"default": "skip"}),
                "negative_conditioning":            ("CONDITIONING", {"tooltip": "➖ Negative prompt conditioning info."}),
                "negative_reference_latent_method": (reference_latent_methods, {"default": "skip"}),
            }
        }

    FUNCTION = "edit_model_reference_method"
    CATEGORY = "Farrenzo's Garbage/Sampling"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("+ Positive", "- Negative")
    DISPLAY_NAME="Edit Model Reference Method"

    def conditioning_set_values(self, conditioning, values={}, append=False):
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            for k in values:
                val = values[k]
                if append:
                    old_val = n[1].get(k, None)
                    if old_val is not None:
                        val = old_val + val

                n[1][k] = val
            c.append(n)

        return c

    def edit_model_reference_method(
        self,
        positive_conditioning,
        positive_reference_latent_method,
        negative_conditioning,
        negative_reference_latent_method
    ):
        # Do the +
        if positive_reference_latent_method == "skip":
            positive_conditioning_output = positive_conditioning
        else:
            positive_conditioning_output = self.conditioning_set_values(
                positive_conditioning, 
                {"reference_latents_method": positive_reference_latent_method}
            )
        # Now do the -
        if negative_reference_latent_method == "skip":
            negative_conditioning_output = negative_conditioning
        else:
            negative_conditioning_output = self.conditioning_set_values(
                negative_conditioning, 
                {"reference_latents_method": negative_reference_latent_method}
            )

        return (positive_conditioning_output, negative_conditioning_output, )

