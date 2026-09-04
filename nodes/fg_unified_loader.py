"""
Unified loader for diffusion models. 
Could've used a subgraph but I was bored.


"""
from nodes import (
    UNETLoader,
    CLIPLoader,
    VAELoader
)

def _spec(cls, key):
    d = cls.INPUT_TYPES()
    return d.get("required", {}).get(key) or d.get("optional", {}).get(key)

class FG_UnifiedModelsLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name":         _spec(UNETLoader, "unet_name"),
                "unet_weight_dtype": _spec(UNETLoader, "weight_dtype"),
                "clip_name":         _spec(CLIPLoader, "clip_name"),
                "clip_type":         _spec(CLIPLoader, "type"),
                "clip_device":       _spec(CLIPLoader, "device"),
                "vae_name":          _spec(VAELoader,  "vae_name"),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_models"
    CATEGORY = "Farrenzo's Garbage/Model Loaders"
    DESCRIPTION = "Unified loader for comfy. Tired of using three nodes to do this."

    def load_models(self, unet_name, unet_weight_dtype, clip_name, clip_type, clip_device, vae_name):
        unet = UNETLoader().load_unet(unet_name, unet_weight_dtype)[0]
        clip = CLIPLoader().load_clip(clip_name, clip_type, clip_device)[0]
        vae  = VAELoader().load_vae(vae_name)[0]
        return (unet, clip, vae)

