from ._fg_helperfunctions import log

from .fg_advanced_ksampler    import FG_Advanced_KSampler
from .fg_CLIP_text_encode     import FG_CLIPTextEncode
from .fg_controlnet           import FG_ApplyControlNet
from .fg_coordinates_box_fill import FG_CoordinatesBoxFill
from .fg_empty_latent         import FG_EmptyLatent
from .fg_image_scale          import FG_ImageScaler
from .fg_lab_color_transfer   import FG_LABColorTransfer
from .fg_load_image           import FG_LoadImage
from .fg_load_vae             import FG_VAELoader
from .fg_lora_loader          import FG_LoraLoader
from .fg_min_max              import FG_MinimumMaximum
from .fg_model_reference      import FG_ModelReferenceLatentMethod
from .fg_purge_vram           import FG_PurgeMemory
from .fg_save_image           import FG_SaveImage
from .fg_show_text            import FG_ShowText
from .fg_telegram_notice      import FG_SendTelegramNotification
from .fg_upscale_model        import FG_ModelImageScaler
from .fg_WD14                 import FG_WD14Tagger

from .fg_anima import AnimaConditioningRegion, ApplyAnimaRegionalConditioningPatch
from .fg_anima_cnet import AnimaLLLiteApply
from .fg_regional_prompt_nodes import (
    MultiLatentComposite,
    MultiAreaConditioning,
    ConditioningUpscale,
    ConditioningStretch,
)

NODE_CLASS_MAPPINGS = {
    "FG_Advanced_KSampler"           : FG_Advanced_KSampler,
    "FG_ApplyControlNet"             : FG_ApplyControlNet,
    "FG_BoxFillwCoordinates"         : FG_CoordinatesBoxFill,
    "FG_CLIPTextEncode"              : FG_CLIPTextEncode,
    "FG_CustomVAELoader"             : FG_VAELoader,
    "FG_DynamicLoraLoader"           : FG_LoraLoader,
    "FG_EmptyLatent"                 : FG_EmptyLatent,
    "FG_ImageScaler"                 : FG_ImageScaler,
    "FG_LABColorTransfer"            : FG_LABColorTransfer,
    "FG_LoadImage"                   : FG_LoadImage,
    "FG_Minimum_Maximum"             : FG_MinimumMaximum,
    "FG_ModelImageScaler"            : FG_ModelImageScaler,
    "FG_ModelReferenceLatentMethod"  : FG_ModelReferenceLatentMethod,
    "FG_PurgeMemory"                 : FG_PurgeMemory,
    "FG_SaveImage"                   : FG_SaveImage,
    "FG_SendTelegramNotification"    : FG_SendTelegramNotification,
    "FG_ShowText"                    : FG_ShowText,
    "FG_WD14Tagger"                  : FG_WD14Tagger,

    "AnimaConditioningRegion": AnimaConditioningRegion,
    "ApplyAnimaRegionalConditioningPatch": ApplyAnimaRegionalConditioningPatch,

    "MultiLatentComposite":  MultiLatentComposite,
    "MultiAreaConditioning": MultiAreaConditioning,
    "ConditioningUpscale":   ConditioningUpscale,
    "ConditioningStretch":   ConditioningStretch,

    "AnimaLLLiteApply": AnimaLLLiteApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FG_Advanced_KSampler"           : "🗑️ Advanced KSampler",
    "FG_ApplyControlNet"             : "🗑️ Apply Advanced ControlNet",
    "FG_BoxFillwCoordinates"         : "🗑️ Coordinate Box Fill",
    "FG_CLIPTextEncode"              : "🗑️ Enhanced CLIP Text Encode",
    "FG_CombinedImageTagger"         : "🗑️ Combined Image Tagger",
    "FG_CustomVAELoader"             : "🗑️ Custom VAE Loader",
    "FG_DynamicLoraLoader"           : "🗑️ Multi-LoRA Loader",
    "FG_EmptyLatent"                 : "🗑️ Advanced Empty Latent",
    "FG_ImageScaler"                 : "🗑️ Image Scaler",
    "FG_KSampler"                    : "🗑️ KSampler for Qwen Image Edit",
    "FG_LABColorTransfer"            : "🗑️ LAB Color Transfer",
    "FG_LoadImage"                   : "🗑️ Load Image",
    "FG_Minimum_Maximum"             : "🗑️ Minimum + Maximum",
    "FG_ModelImageScaler"            : "🗑️ Image Scale with Model",
    "FG_ModelReferenceLatentMethod"  : "🗑️ Edit Model Reference Method",
    "FG_PurgeMemory"                 : "🗑️ Purge Memory",
    "FG_SaveImage"                   : "🗑️ Save Image",
    "FG_SendTelegramNotification"    : "🗑️ Send Telegram Notification",
    "FG_ShowText"                    : "🗑️ Show Text",
    "FG_WD14Tagger"                  : "🗑️ WD14 Tagger (Booru Tags)",

    "AnimaConditioningRegion":             "⚙️ Anima Conditioning Region",
    "ApplyAnimaRegionalConditioningPatch": "⚙️ Apply Anima Regional Conditioning Patch",

    "AnimaLLLiteApply": "⚙️ Apply Anima ControlNet-LLLite",

    "MultiLatentComposite":  "⚙️ Multi Latent Composite",
    "MultiAreaConditioning": "⚙️ Multi Area Conditioning",
    "ConditioningUpscale":   "⚙️ Conditioning Upscale",
    "ConditioningStretch":   "⚙️ Conditioning Stretch",
}


# Some required external libraries
try:
    # Ollama
    from .fg_ollama import (
        OllamaOptionsV2,
        OllamaConnectivityV2,
        OllamaGenerateV2,
        OllamaSaveContext,
        OllamaLoadContext,
        OllamaChat,
    )
    NODE_CLASS_MAPPINGS = {
        **NODE_CLASS_MAPPINGS,
        "OllamaOptionsV2"     : OllamaOptionsV2,
        "OllamaConnectivityV2": OllamaConnectivityV2,
        "OllamaGenerateV2"    : OllamaGenerateV2,
        "OllamaSaveContext"   : OllamaSaveContext,
        "OllamaLoadContext"   : OllamaLoadContext,
        "OllamaChat"          : OllamaChat,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        **NODE_DISPLAY_NAME_MAPPINGS,
        "OllamaOptionsV2"                : "⚙️ Ollama Options",
        "OllamaConnectivityV2"           : "⚙️ Ollama Connectivity",
        "OllamaGenerateV2"               : "⚙️ Ollama Generate",
        "OllamaSaveContext"              : "⚙️ Ollama Save Context",
        "OllamaLoadContext"              : "⚙️ Ollama Load Context",
        "OllamaChat"                     : "⚙️ Ollama Chat",
    }
except ImportError as e:
    log(
        message=f"Unable to import Ollama, try installing it with PIP\nOllama nodes will be unavailable.\n{e}",
        message_type="error"
    )