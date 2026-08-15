"""
Aspect Ratio Latent Image Node for ComfyUI
Combines aspect ratio selection with empty latent generation.

┌─────────────────────────────────────┐
│      Aspect Ratio Latent Image      │
├─────────────────────────────────────┤
│ <▼DROPDOWN_1>                       │
│ ○ Manual           Latent Image  ○  │
│ ○ Square                  Width  ○  │
│ ○ Horizontal             Height  ○  │
│ ○ Vertical                          │
│                                     │
│ <▼DROPDOWN_2>                       │
│ - List populated from selection     │
│                                     │
│ <→ TEXT INPUT_Width>                │
│ - Auto Populated if not manual      │
│                                     │
│ <→ TEXT INPUT_Height>               │
│ - Auto Populated if not manual      │
│                                     │
│ <→ TEXT INPUT_Batch Size>           │
│ - Default 1, required               │
│                                     │
└─────────────────────────────────────┘

"""

from ._fg_helperfunctions import (
    log, 
    MODEL_TYPES, 
    ASPECT_RATIOS,
    generate_latent_image_data
)

MAX_RESOLUTION = 16384

class FG_EmptyLatent:
    CSS_PATH = "css/custom.css"
    
    # Flatten all options for initial widget setup
    ALL_DIMENSIONS = ["Manual"]
    for orientation, dims in ASPECT_RATIOS.items():
        ALL_DIMENSIONS.extend(list(dims.keys()))

    def __init__(self):
        self.NODE_NAME = "Advanced Empty Latent"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "orientation": (["Manual", "Square", "Horizontal", "Vertical"], {"default": "Square", "tooltip": 'If not "Manual", values in width and height will be ignored.'}),
                "dimensions": (cls.ALL_DIMENSIONS, {"default": "1:1 | 64 | 512×512", "tooltip": "Ratio | Divisible by | dimensions. Do not select any setting with X or 8 if running for Flux."}),
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Width in pixels. Auto-populated unless Manual mode."}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Height in pixels. Auto-populated unless Manual mode."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 40, "tooltip": "Number of latent images in the batch."}),
            },
            "optional": {
                "model_type" : (list(MODEL_TYPES.keys()), {"default": "SDXL", "tooltip": "For an empty latent, SDXL & FLUX are different."}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("Latent", "Width", "Height")
    OUTPUT_TOOLTIPS = ("The empty latent image batch.", "Width in pixels.", "Height in pixels.")
    FUNCTION = "generate"
    CATEGORY = "Farrenzo's Garbage/Utils"
    DESCRIPTION = "Create empty latent images with preset or custom aspect ratios."

    def generate(self, orientation, dimensions, width, height, batch_size=1, model_type="SDXL"):
        # In Manual mode, use the width/height inputs directly
        # Otherwise, look up from the dimensions selection
        if orientation != "Manual" and dimensions != "Manual":
            # Find the dimensions in our lookup
            for orient, dims in ASPECT_RATIOS.items():
                if dimensions in dims:
                    width, height = dims[dimensions]
                    break

        # Generate the latent
        latent_info, latent = generate_latent_image_data(width=width, height=height, batch_size = batch_size, model_type=model_type)
        log(f"{self.NODE_NAME}: Generated an {latent_info} latent of {width}*{height}")
        return (latent, width, height, )
