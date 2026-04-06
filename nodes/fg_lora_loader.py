"""
┌─────────────────────────────────────┐
│      Aspect Ratio Latent Image      │
├─────────────────────────────────────┤
│ ○ Model                    Model  ○ │
│ ○ Clip                      Clip  ○ │
│                                     │
│          + Add LoRA Button          │
│                                     │
└─────────────────────────────────────┘

"""
import json
import comfy.sd
import comfy.utils
import folder_paths
from ._fg_helperfunctions import log
from .. import LORA_INDEX

global_description = """
Dynamic LoRA stacker with optional CLIP entry.
 - Can be used with SDXL models. (CLIP required)
 - FLUX models. (CLIP optional)
"""

class FG_LoraLoader:
    """LoRA loader with dynamic add/remove buttons. CLIP input is optional."""
    
    def __init__(self):
        self.loaded_loras = {}
        self.NODE_NAME = "Multi-LoRA Loader"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "🤖 Model: Diffusion Model input used for generation"}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "📒 Clip: used for text encoding and conditional generation. This is optional."}),
            },
            "hidden": {
                "lora_stack": ("STRING", {"default": "[]"}),  # JSON array of LoRA configs
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("Model", "Clip")
    FUNCTION = "load_loras"
    CATEGORY = "Farrenzo's Garbage/Loaders"
    DESCRIPTION = global_description

    def load_loras(self, model, clip=None, **kwargs):
        self.loaded_loras = {}

        try:
            loras = json.loads(kwargs.get("lora_stack"))
        except json.JSONDecodeError:
            loras = []

        current_model = model
        current_clip = clip  # None if not connected

        for lora_config in loras:
            name = lora_config.get("name")
            if not name:
                continue

            strength_model = float(lora_config.get("strength_model", 1.0))
            strength_clip = float(lora_config.get("strength_clip", 1.0)) if current_clip is not None else 0

            if strength_model == 0 and strength_clip == 0:
                continue

            lora_path = folder_paths.get_full_path_or_raise("loras", name)


            if lora_path in self.loaded_loras:
                lora = self.loaded_loras[lora_path]
            else:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_loras[lora_path] = lora

            current_model, current_clip = comfy.sd.load_lora_for_models(
                current_model, current_clip, lora, strength_model, strength_clip
            )

        logger = True
        while logger:
            if len(self.loaded_loras) > 0:
                log(f"{self.NODE_NAME}: Total LoRA's: {len(self.loaded_loras)}")
                for i in self.loaded_loras:
                    log(f"{self.NODE_NAME}:Loaded: {i[i.rfind('\\') + 1:]}")
            else:
                log(f"{self.NODE_NAME}: No LoRA's loaded.")
            logger = False

        return (current_model, current_clip)
