from ._fg_helperfunctions    import log
from .coordinates_box_fill   import CoordinatesBoxFill
from .fg_advanced_ksampler   import FG_Advanced_KSampler
from .fg_CLIP_text_encode    import FG_CLIPTextEncode
from .fg_controlnet          import FG_ApplyControlNet
from .fg_empty_latent        import FG_EmptyLatent
from .fg_image_scale         import FG_ImageScaler
from .fg_lab_color_transfer  import FG_LABColorTransfer
from .fg_load_image          import FG_LoadImage
from .fg_load_vae            import FG_VAELoader
from .fg_lora_loader         import FG_LoraLoader
from .fg_min_max             import MinimumMaximum
from .fg_model_reference     import FG_ModelReferenceLatentMethod
from .fg_purge_vram          import FG_PurgeMemory
from .fg_save_image          import FG_SaveImage
from .fg_show_text           import ShowText
from .fg_telegram_notice     import SendTelegramNotification
from .fg_WD14                import WD14Tagger
from .fg_upscale_model       import FG_ModelImageScaler

from .fg_ollama import (
    OllamaOptionsV2,
    OllamaConnectivityV2,
    OllamaGenerateV2,
    OllamaSaveContext,
    OllamaLoadContext,
    OllamaChat,
)

NODE_CLASS_MAPPINGS = {
    "FG_Advanced_KSampler"           : FG_Advanced_KSampler,
    "FG_ApplyControlNet"             : FG_ApplyControlNet,
    "FG_BoxFillwCoordinates"         : CoordinatesBoxFill,
    "FG_CLIPTextEncode"              : FG_CLIPTextEncode,
    "FG_CustomVAELoader"             : FG_VAELoader,
    "FG_DynamicLoraLoader"           : FG_LoraLoader,
    "FG_EmptyLatent"                 : FG_EmptyLatent,
    "FG_ImageScaler"                 : FG_ImageScaler,
    "FG_LABColorTransfer"            : FG_LABColorTransfer,
    "FG_LoadImage"                   : FG_LoadImage,
    "FG_Minimum_Maximum"             : MinimumMaximum,
    "FG_ModelReferenceLatentMethod"  : FG_ModelReferenceLatentMethod,
    "FG_PurgeMemory"                 : FG_PurgeMemory,
    "FG_SaveImage"                   : FG_SaveImage,
    "FG_SendTelegramNotification"    : SendTelegramNotification,
    "FG_ShowText"                    : ShowText,
    "FG_WD14Tagger"                  : WD14Tagger,
    "FG_ModelImageScaler"            : FG_ModelImageScaler,

    # Ollama
    "OllamaOptionsV2"     : OllamaOptionsV2,
    "OllamaConnectivityV2": OllamaConnectivityV2,
    "OllamaGenerateV2"    : OllamaGenerateV2,
    "OllamaSaveContext"   : OllamaSaveContext,
    "OllamaLoadContext"   : OllamaLoadContext,
    "OllamaChat"          : OllamaChat,

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
    "FG_ModelReferenceLatentMethod"  : "🗑️ Edit Model Reference Method",
    "FG_PurgeMemory"                 : "🗑️ Purge Memory",
    "FG_SaveImage"                   : "🗑️ Save Image",
    "FG_SendTelegramNotification"    : "🗑️ Send Telegram Notification",
    "FG_ShowText"                    : "🗑️ Show Text",
    "FG_WD14Tagger"                  : "🗑️ WD14 Tagger (Booru Tags)",
    "FG_ModelImageScaler"            : "🗑️ Image Scale with Model",

    # Ollama
    "OllamaOptionsV2"                : "Ollama Options",
    "OllamaConnectivityV2"           : "Ollama Connectivity",
    "OllamaGenerateV2"               : "Ollama Generate",
    "OllamaSaveContext"              : "Ollama Save Context",
    "OllamaLoadContext"              : "Ollama Load Context",
    "OllamaChat"                     : "Ollama Chat",
}