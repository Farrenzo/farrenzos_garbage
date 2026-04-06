"""
Image Scaler
┌─────────────────────────────────────┐
│       Image Scaler with Model       │
├─────────────────────────────────────┤
│ ○ Image                  Latent  ○  │
│ ○ VAE                               │
│                                     │
│ <→ INPUT> Width  Default = 0        │
│ <→ INPUT> Height Default = 0        │
│ <→ INPUT> Tile Size Default = 512   │
│ <→ INPUT> Overlap Default = 512     │
│ <→ INPUT> Temporal Size = 64        │
│ <→ INPUT> Temporal Overlap = 64     │
│ <▼ DROPDOWN> Crop                   │
│ <▼ DROPDOWN> Scaling Method         │
│ <▼ DROPDOWN> UpScale Model          │
│                                     │
└─────────────────────────────────────┘

Merged three nodes into one.
"""

import torch
import folder_paths
import comfy.utils as c_utils
import comfy.model_management
from ._fg_helperfunctions import log
from spandrel import ModelLoader, ImageModelDescriptor

class FG_ModelImageScaler:

    def __init__(self):
        self.NODE_NAME = "Model Image Scaler"
        self.device = comfy.model_management.get_torch_device()

    @classmethod
    def INPUT_TYPES(self):
        scaling_methods = ["lanczos", "bilinear", "bicubic", "nearest-exact", "area"]
        return {
            "required": {
                "image": ("IMAGE",),
                "vae"  : ("VAE", {"tooltip": "🔣 VAE: Used to encode the reference images into latent the model understands."}),
                "desired_width"    : ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1, "tooltip": "The width you want the image to turn out to be. Zero will have no change in original image."}),
                "desired_height"   : ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1, "tooltip": "The height you want the image to turn out to be. Zero will have no change in original image."}),
                "tile_size"        : ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64, "advanced": True}),
                "overlap"          : ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32, "advanced": True}),
                "temporal_size"    : ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to encode at a time.", "advanced": True}),
                "temporal_overlap" : ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to overlap.", "advanced": True}),
                "crop"             : (["center", "disabled"], {"default": "center"}),
                "scaling_method"   : (scaling_methods, {"default": "lanczos"}),
                "upscale_model"    : (folder_paths.get_filename_list("upscale_models"), {"tooltip": "Select model to be used to upscale."}),
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("Latent", )
    FUNCTION = "scale_with_model"
    CATEGORY = "Farrenzo's Garbage/Utils"

    def _load_model(self, model_path):
        if hasattr(self, '_cached_model_path') and self._cached_model_path == model_path:
            return self._cached_model
        model_path = folder_paths.get_full_path_or_raise("upscale_models", model_path)
        sd = c_utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = c_utils.state_dict_prefix_replace(sd, {"module.":""})
        model = ModelLoader().load_from_state_dict(sd).eval()

        if not isinstance(model, ImageModelDescriptor):
            raise Exception("Upscale model must be a single-image model.")
        
        self._cached_model_path = model_path
        self._cached_model = model
        return model

    def _upscale_w_model(self, model, pic):
        memory_required = comfy.model_management.module_size(model.model)
        memory_required += (512 * 512 * 3) * pic.element_size() * max(model.scale, 1.0) * 384.0
        # The 384.0 is an estimate of how much some of these models take,
        # TODO: make it more accurate
        memory_required += pic.nelement() * pic.element_size()
        comfy.model_management.free_memory(memory_required, self.device)
        model.to(self.device)
        in_img = pic.movedim(-1,-3).to(self.device)

        tile = 512
        overlap = 32

        oom = True
        try:
            while oom:
                try:
                    steps = in_img.shape[0] * c_utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                    pbar = c_utils.ProgressBar(steps)
                    s = c_utils.tiled_scale(in_img, lambda a: model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=model.scale, pbar=pbar)
                    oom = False
                except Exception as e:
                    comfy.model_management.raise_non_oom(e)
                    tile //= 2
                    if tile < 128:
                        raise e
        finally:
            model.to("cpu")
        model_scaled_image_v1 = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
        return model_scaled_image_v1

    def _upscale_w_dimensions(self, image, width, height, upscale_method, crop = "center"):
        if width == 0 and height == 0:
            return image

        og_height, og_width = image.shape[1], image.shape[2]
        if width == 0:
            width = max(1, round(og_width * height / og_height))
        elif height == 0:
            height = max(1, round(og_height * width / og_width))

        samples = image.movedim(-1, 1)
        s = c_utils.common_upscale(samples, width, height, upscale_method, crop)
        scaled_image = s.movedim(1, -1)

        return scaled_image

    def scale_with_model(
        self,
        image,
        vae,
        tile_size,
        overlap,
        temporal_size,
        temporal_overlap,
        scaling_method,
        desired_width,
        desired_height,
        upscale_model,
        crop = "center"
    ):
        # Load model
        model = self._load_model(upscale_model)
        log(f"{self.NODE_NAME}: Loaded model ...next is model upscale.")
        # Upscale round 1
        model_scaled_image_v1 = self._upscale_w_model(model, image)
        log(f"{self.NODE_NAME}: Done with model upscale ...now scaling image to desired dimensions.")
        # Upscale round 2
        upscaled_image_v2 = self._upscale_w_dimensions(
            image          = model_scaled_image_v1,
            width          = desired_width,
            height         = desired_height,
            upscale_method = scaling_method,
            crop           = crop
        )
        # VAE encode tiled
        tiled_vae = vae.encode_tiled(upscaled_image_v2, tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)

        return ({"samples": tiled_vae}, )

