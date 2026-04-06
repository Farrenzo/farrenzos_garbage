"""
Load and apply controlnet all in one.
Currently, it does nothing with a mask.
"""
import folder_paths
import comfy.controlnet
from ._fg_helperfunctions import (
    log,
    tensor2pil,
    generate_latent_image_data,
    MODEL_TYPES
)


class FG_ApplyControlNet:

    def __init__(self):
        self.NODE_NAME = "Apply Advanced ControlNet"

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "image": ("IMAGE", ),
                "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
            "optional": {
                "mask": ("MASK",),
                "vae": ("VAE", ),
                "base_model" : (list(MODEL_TYPES.keys()), {"default": "SDXL", "tooltip": "For an empty latent, SDXL & FLUX are different."}),
                "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
            }
    }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING", "LATENT")
    RETURN_NAMES = ("Positive +", "Negative -", "Latent Noise Mask", )
    CATEGORY = "Farrenzo's Garbage/controlnet"
    FUNCTION = "apply_controlnet"
    SEARCH_ALIASES = ["controlnet", "apply controlnet", "use controlnet", "control net"]
    DESCRIPTION = "ControlNet Integrated Loader & Applicator."

    def apply_controlnet(
        self,
        positive,
        negative,
        image,
        strength,
        start_percent,
        end_percent,
        control_net_name,
        base_model   = "SDXL",
        grow_mask_by = 6,
        vae          = None,
        mask         = None,
        extra_concat = []
    ):
        if strength == 0:
            log(f"{self.NODE_NAME}: Controlnet with strength of zero not applied.")
            return (positive, negative)

        controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
        control_net = comfy.controlnet.load_controlnet(controlnet_path)
        if control_net is None:
            raise RuntimeError("ERROR: Controlnet file is invalid and does not contain a valid controlnet model.")

        image_width, image_height = tensor2pil(image[0]).size
        if vae is None:
            latent_info, latent = generate_latent_image_data(width=image_width, height=image_height, model_type=base_model)
            log(f"{self.NODE_NAME}: No VAE to decode image. Generated an {latent_info} latent of {image_width}*{image_height}")
        elif vae:
            if not mask:
                latent_info, latent = generate_latent_image_data(width=image_width, height=image_height, vae = vae, image = image)
                log(f"{self.NODE_NAME}: Found VAE, but no mask, only encoding image into latent.")
            if mask and len(mask) > 0:
                latent_info, latent = generate_latent_image_data(
                    width           = image_width,
                    height          = image_height,
                    model_type      = base_model,
                    vae             = vae,
                    mask            = mask,
                    image           = image,
                    mask_growth_val = grow_mask_by
                )
                log(f"{self.NODE_NAME}: Found mask and VAE, encoding latent for inpainting.")

        control_hint = image.movedim(-1,1)
        cnets = {}
        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent), vae=vae, extra_concat=extra_concat)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1], latent)


