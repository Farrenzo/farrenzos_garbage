"""
Image Scaler
┌─────────────────────────────────────┐
│            Image Scaler             │
├─────────────────────────────────────┤
│ ○ Image                   Image  ○  │
│ ○ Mask                     Mask  ○  │
│ ○ VAE                     Width  ○  │
│                          Height  ○  │
│                             VAE  ○  │
│ <▼ DROPDOWN> Fit                    │
│ <↔ BOOLEAN>  Rounding               │
│ <▼ DROPDOWN> Scaling Method         │
│ <▼ DROPDOWN> Round to Multiple      │
│ <→ INPUT> Width  Default = 0        │
│ <→ INPUT> Height Default = 0        │
│ <→ INPUT> Background Color          │
│ Default = #FFFFFF                   │
│                                     │
└─────────────────────────────────────┘

Image:
Although can accept batches, will work best on only one image at a time. If you
insist on using a batch, make sure the first image is the desired Width*Height as
the rest will be that same width height. Also, vae will only encode the first image.

Optional Mask, VAE
Output: Image, Width, Height, -- (optional) Mask, Latent

"""

import math
import torch
import comfy.model_management
from ._fg_helperfunctions import (
    log,
    tensor2pil,
    pil2tensor,
    image2mask,
    unpack_images,
    unpack_masks,
    fit_resize_image,
    generate_latent_image_data,
    MODEL_TYPES,
    SCALING_METHODS
)

class FG_ImageScaler:

    def __init__(self):
        self.NODE_NAME = "Image Scale"
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(self):
        fit_mode = ["crop", "fill", "letterbox"]
        multiple_list = ["8", "16", "32", "64", "128", "256", "512", "None"]
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),  #
                "vae": ("VAE", {"tooltip": "🔣 VAE: Used to encode the reference images into latent the model understands."}),
                "rounding"         : ("BOOLEAN", {"default": True, "label_on": "Round Up", "label_off": "Round Down"}),
                "grow_mask_by"     : ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                "fit"              : (fit_mode, {"default": "fill"}),
                "scaling_method"   : (list(SCALING_METHODS.keys()), {"default": "lanczos"}),
                "round_to_multiple": (multiple_list, {"default": "64"}),
                "desired_width"    : ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "desired_height"   : ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "background_color" : ("STRING", {"default": "#FFFFFF"}),  # Background color
                "base_model"       : (list(MODEL_TYPES.keys()), {"default": "SDXL", "tooltip": "For an empty latent, SDXL & FLUX are different."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "LATENT", "INT", "INT",)
    RETURN_NAMES = ("Image", "Mask", "Latent", "Width", "Height",)
    FUNCTION = "scale_image"
    CATEGORY = "Farrenzo's Garbage/Utils"

    def _calculate_WH(self, width, height, multiple=8, rounding=True): ## newly rounded width & height
        if rounding:
            # Round UP to the nearest multiple
            width  = math.ceil(width / multiple) * multiple
            height = math.ceil(height / multiple) * multiple
        else:
            # Round DOWN to the nearest multiple
            width  = (width // multiple) * multiple
            height = (height // multiple) * multiple

        return int(width), int(height)
    
    def scale_image(
        self,
        image,
        fit,
        scaling_method,
        rounding          = True,
        vae               = None,
        mask              = None,
        grow_mask_by      = 6,
        desired_width     = 0,
        desired_height    = 0,
        round_to_multiple = 64,
        background_color  = "#FFFFFF",
        base_model        = "SDXL"
    ):
        target_width, target_height = None, None
        resize_sampler = SCALING_METHODS[scaling_method]
        latent = None
        # Define the images
        output_images = []
        output_masks = []
        round_to_multiple_val = None if round_to_multiple == "None" else int(round_to_multiple)

        # Process the images
        unpacked_images, og_width, og_height = unpack_images(images= image)
        # Process the masks
        unpacked_masks = []
        if mask is not None:
            unpacked_masks, mask_width, mask_height = unpack_masks(masks = mask)
            if mask_width == 0 or mask_height == 0:
                log(f"{self.NODE_NAME}: Input mask is empty, ignoring it.", message_type="warning")
            else:
                if (og_width != mask_width) or (og_height != mask_height):
                    log(f"{self.NODE_NAME}: First mask doesn't match first image. Skipping mask.", message_type="warning")

        # Get target dimensions
        semantics = "up" if rounding else "down"
        has_dims   = desired_width > 0 and desired_height > 0
        valid_dims = desired_width >= 512 and desired_height >= 512
        has_round  = round_to_multiple_val is not None
        scenario   = (has_dims, valid_dims, has_round)
        actions    = {
            # 1. Dims provided but too small
            (True,  False, False): ("no_scale",  "Dimensions below 512, using original."),
            (True,  False, True) : ("no_scale",  "Dimensions below 512, using original."),
            # 2. Valid dims, no rounding conflict
            (True,  True,  False): ("use_dims",  f"Scaling to {desired_width}x{desired_height}."),
            # 3. Valid dims but rounding also set — conflicting
            (True,  True,  True) : ("no_scale",  "Conflicting options: dimensions AND rounding set. Using original."),
            # 4. No dims, no rounding
            (False, False, False): ("no_scale",  "No scaling requested."),
            # 5. No dims, rounding selected
            (False, False, True) : ("round",     f"Rounding {semantics} to nearest multiple of {round_to_multiple_val}."),
        }
        action, message = actions[scenario]
        log(f"{self.NODE_NAME}: {message}")
        if action == "use_dims":
            target_width, target_height = desired_width, desired_height
            scale = True
        elif action == "round":
            target_width, target_height = self._calculate_WH(og_width, og_height, multiple=round_to_multiple_val, rounding=rounding)
            scale = True
        else:
            target_width, target_height = og_width, og_height
            scale = False

        # Scale
        if not scale:
            output_images = unpacked_images
            if len(unpacked_masks) > 0:
                output_masks = unpacked_masks
        else:
            for i in unpacked_images:
                _im = tensor2pil(i).convert('RGB')
                _im = fit_resize_image(_im, target_width, target_height, fit, resize_sampler, background_color)
                output_images += [pil2tensor(_im)]
            
            if len(unpacked_masks) > 0:
                for m in unpacked_masks:
                    _ma = tensor2pil(m).convert('L')
                    _ma = fit_resize_image(_ma, target_width, target_height, fit, resize_sampler).convert('L')
                    output_masks += [image2mask(_ma)]

        if latent is None:
            if vae is None:
                latent_info, latent = generate_latent_image_data(width=target_width, height=target_height, model_type = base_model)
                log(f"{self.NODE_NAME}: No VAE to decode image. Generated an {latent_info} latent of {target_width}*{target_height}")
            elif vae and len(output_masks) == 0:
                latent_info, latent = generate_latent_image_data(width=target_width, height=target_height, vae = vae, image = output_images[0])
                log(f"{self.NODE_NAME}: Found VAE, but no mask, only encoding image into latent.")
            elif vae and len(output_masks) > 0:
                latent_info, latent = generate_latent_image_data(
                    width      = target_width,
                    height     = target_height,
                    batch_size = len(output_images),
                    model_type = base_model,
                    vae        = vae,
                    image      = output_images[0],
                    mask       = output_masks[0],
                    mask_growth_val = grow_mask_by
                )
                log(f"{self.NODE_NAME}: Found mask and VAE, encoding latent for inpainting.")


        total_images = len(output_images)
        total_masks = len(output_masks)
        final_outputs = (
            torch.cat(output_images, dim=0),
            torch.cat(output_masks, dim=0) if total_masks > 0 else None,
            latent,
            target_width,
            target_height,
        )

        log(
            f"{self.NODE_NAME} Processed {total_images} image, {total_masks} mask & an {latent_info} latent.",
            message_type='finish'
        )
        return final_outputs
