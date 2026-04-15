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

from ._fg_helperfunctions import MODEL_TYPES, log, generate_latent_image_data

MAX_RESOLUTION = 16384

class FG_EmptyLatent:
    
    ASPECT_RATIOS = {
        "Square": {
            "1:1 | 64 | 512×512":   (512, 512),
            "1:1 | 64 | 576×576":   (576, 576),
            "1:1 | 64 | 640×640":   (640, 640),
            "1:1 | 64 | 704×704":   (704, 704),
            "1:1 | 64 | 768×768":   (768, 768),
            "1:1 | 64 | 832×832":   (832, 832),
            "1:1 | 64 | 896×896":   (896, 896),
            "1:1 | 64 | 960×960":   (960, 960),
            "1:1 | 64 | 1024×1024": (1024, 1024),
            "1:1 | 64 | 1088×1088": (1088, 1088),
            "1:1 | 64 | 1152×1152": (1152, 1152),
            "1:1 | 64 | 1216×1216": (1216, 1216),
            "1:1 | 64 | 1280×1280": (1280, 1280),
            "1:1 | 64 | 1344×1344": (1344, 1344),
        },
        "Horizontal": {
            "1.42:1 | 64 | 1344×960"  :(1344, 960),
            "1.42:1 | 64 | 1088×768"  :(1088, 768),
            "1.85:1 | 64 | 960×512"   :(960, 512),
            "1.85:1 | X | 1024×540"   :(1024, 540),
            "1.85:1 | 64 | 1088×576"  :(1088, 576),
            "1.85:1 | 64 | 1216×640"  :(1216, 640),
            "1.85:1 | 64 | 1536×832"  :(1536, 832),
            "1.85:1 | 64 | 1792×960"  :(1792, 960),
            "1.85:1 | 64 | 1920×1024" :(1920, 1024),
            "2:1 | 64 | 1408×704":(1408, 704),
            "3:1 | 64 | 1536×512":(1536, 512),
            "3:1 | 64 | 1728×576":(1728, 576),
            "3:1 | 64 | 1920×640":(1920, 640),
            "3:1 | 64 | 2304×768":(2304, 768),
            "3:2 | 64 | 768×512":(768, 512),
            "3:2 | 64 | 1152×768":(1152, 768),
            "3:2 | 64 | 1344×896":(1344, 896),
            "3:2 | 64 | 1536×1024":(1536, 1024),
            "3:2 | 64 | 1728×1152":(1728, 1152),
            "4:1 | 64 | 2048×512":(2048, 512),
            "4:1 | 64 | 2304×576":(2304, 576),
            "4:1 | 64 | 2560×640":(2560, 640),
            "4:1 | 64 | 2816×704":(2816, 704),
            "4:3 | 64 | 768×576":(768, 576),
            "4:3 | 64 | 1024×768":(1024, 768),
            "4:3 | 64 | 1280×960":(1280, 960),
            "4:3 | 64 | 1472×1088":(1472, 1088),
            "4:3 | 64 | 1536×1152":(1536, 1152),
            "5:2 | 64 | 1600×640":(1600, 640),
            "5:3 | 64 | 1280×768":(1280, 768),
            "5:4 | 64 | 960×768":(960, 768),
            "5:4 | 64 | 1280×1024":(1280, 1024),
            "5:4 | 64 | 1536×1216":(1536, 1216),
            "5:4 | 64 | 1600×1280":(1600, 1280),
            "7:4 | 64 | 1344×768":(1344, 768),
            "9:7 | 64 | 1152×896":(1152, 896),
            "12:5 | 64 | 1536×640":(1536, 640),
            "16:9 | 64 | 1024×576":(1024, 576),
            "16:9 | 64 | 1152×640":(1152, 640),
            "16:9 | 64 | 1344×768":(1344, 768),
            "16:9 | 64 | 1472×832":(1472, 832),
            "16:9 | 64 | 1536×896":(1536, 896),
            "16:9 | 64 | 1728×960":(1728, 960),
            "16:9 | 8 | 1920×1080":(1920, 1080),
            "16:15 | 64 | 1024×960":(1024, 960),
            "17:14 | 64 | 1088×896":(1088, 896),
            "17:15 | 64 | 1088×960":(1088, 960),
            "18:13 | 64 | 1152×832":(1152, 832),
            "19:13 | 64 | 1216×832":(1216, 832),
            "21:11 | 64 | 1344×704":(1344, 704),
            "21:9 | 64 | 1856×768":(1856, 768),
            "21:9 | 64 | 2176×896":(2176, 896),
            "23:11 | 64 | 1472×704":(1472, 704),
            "26:9 | 64 | 1664×576":(1664, 576),
            "32:9 | 64 | 1856×512":(1856, 512),
            "32:9 | 64 | 2048×576":(2048, 576),
            "32:9 | 64 | 2240×640":(2240, 640),
            "32:9 | 64 | 2496×704":(2496, 704),
        },
        "Vertical": {
            "1:1.42 | 64 | 960×1344":(960, 1344),
            "1:1.42 | 64 | 768×1088":(768, 1088),
            "1:1.85 | 64 | 512×960":(512, 960),
            "1:1.85 | X | 540×1024":(540, 1024),
            "1:1.85 | 64 | 576×1088":(576, 1088),
            "1:1.85 | 64 | 640×1216":(640, 1216),
            "1:1.85 | 64 | 832×1536":(832, 1536),
            "1:1.85 | 64 | 960×1792":(960, 1792),
            "1:1.85 | 64 | 1024×1920":(1024, 1920),
            "1:2 | 64 | 704×1408":(704, 1408),
            "2:3 | 64 | 512×768":(512, 768),
            "2:3 | 64 | 768×1152":(768, 1152),
            "2:3 | 64 | 896×1344":(896, 1344),
            "2:3 | 64 | 1024×1536":(1024, 1536),
            "2:3 | 64 | 1152×1728":(1152, 1728),
            "3:4 | 64 | 576×76":(576, 768),
            "3:4 | 64 | 768×1024":(768, 1024),
            "3:4 | 64 | 960×1280":(960, 1280),
            "3:4 | 64 | 1088×1472":(1088, 1472),
            "3:4 | 64 | 1152×1536":(1152, 1536),
            "3:5 | 64 | 768×1280":(768, 1280),
            "4:7 | 64 | 768×1344":(768, 1344),
            "4:5 | 64 | 768×960":(768, 960),
            "4:5 | 64 | 1024×1280":(1024, 1280),
            "4:5 | 64 | 1216×1536":(1216, 1536),
            "4:5 | 64 | 1280×1600":(1280, 1600),
            "7:9 | 64 | 896×1152 ":(896, 1152),
            "9:16 | 64 | 576×1024":(576, 1024),
            "9:16 | 64 | 896×1536":(896, 1536),
            "9:16 | 64 | 832×1472":(832, 1472),
            "9:16 | 64 | 768×1344":(768, 1344),
            "9:16 | 64 | 640×1152":(640, 1152),
            "9:16 | 64 | 960×1728":(960, 1728),
            "9:16 | 8 | 1080×1920":(1080, 1920),
            "9:21 | 64 | 768×1856":(768, 1856),
            "9:21 | 64 | 896×2176":(896, 2176),
            "9:32 | 64 | 512×1856":(512, 1856),
            "9:32 | 64 | 576×2048":(576, 2048),
            "9:32 | 64 | 640×2240":(640, 2240),
            "9:32 | 64 | 704×2496":(704, 2496),
            "11:21 | 64 | 704×1344":(704, 1344),
            "13:19 | 64 | 832×1216":(832, 1216),
            "13:18 | 64 | 832×1152":(832, 1152),
            "14:17 | 64 | 896×1088":(896, 1088),
            "15:16 | 64 | 960×1024":(960, 1024),
            "15:17 | 64 | 960×1088":(960, 1088),
        },
    }
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
            for orient, dims in self.ASPECT_RATIOS.items():
                if dimensions in dims:
                    width, height = dims[dimensions]
                    break

        # Generate the latent
        latent_info, latent = generate_latent_image_data(width=width, height=height, batch_size = batch_size, model_type=model_type)
        log(f"{self.NODE_NAME}: Generated an {latent_info} latent of {width}*{height}")
        return (latent, width, height, )
