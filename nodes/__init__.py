from ._fg_helperfunctions import log
from .fg_intelCode import install_xpu_patch, start_keepalive

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
from .fg_save_video           import FG_SaveVideo
from .fg_show_text            import FG_ShowText
from .fg_telegram_notice      import FG_SendTelegramNotification
from .fg_upscale_model        import FG_ModelImageScaler
from .fg_WD14                 import FG_WD14Tagger
from .fg_xpu_guard            import FG_XPUGuard
from .fg_unified_loader       import FG_UnifiedModelsLoader

from .fg_anima.anima_controlnet_nodes import AnimaLLLiteApply
from .fg_anima.anima_regional_prompt_nodes import AnimaConditioningRegion, ApplyAnimaRegionalConditioningPatch
from .fg_anima.anima_ipadapter_nodes import (
    AnimaIPAdapterLoader,
    AnimaIPAdapterApply,
    AnimaSiglipeEncodeImage,
    AnimaImageEmbLoader,
)

from .fg_krea2_rebalance import ConditioningKrea2Rebalance
from .fg_regional_prompt_nodes import (
    MultiLatentComposite,
    MultiAreaConditioning,
    ConditioningUpscale,
    ConditioningStretch,
)

from .fg_SCAIL2 import (
    SCAIL2EasyConfig,
    SCAIL2AutoVideo,
    SCAIL2RunInfo,
)

from .fg_minimax.fg_minimax_autochain import (
MiniMaxH3AutoChainMotionContext,
MiniMaxH3AutoChainMotionContextTrim,
MiniMaxH3AutoChainSaveLatent,
MiniMaxH3AutoChainLoadLatent,
MiniMaxH3AutoChainAudio,
MiniMaxH3AutoChain,
MiniMaxH3AutoChainFrameReference,
)
from .fg_minimax.fg_minimax_h3 import FG_MiniMaxH3_Conditioner

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
    "FG_SaveVideo"                   : FG_SaveVideo,
    "FG_SendTelegramNotification"    : FG_SendTelegramNotification,
    "FG_ShowText"                    : FG_ShowText,
    "FG_WD14Tagger"                  : FG_WD14Tagger,
    "FG_XPUGuard"                    : FG_XPUGuard,
    "FG_MiniMaxH3_Conditioner"       : FG_MiniMaxH3_Conditioner,
    "FG_UnifiedModelsLoader"         : FG_UnifiedModelsLoader,

    # Experimental Anima nodes
    "AnimaConditioningRegion": AnimaConditioningRegion,
    "ApplyAnimaRegionalConditioningPatch": ApplyAnimaRegionalConditioningPatch,
    "AnimaLLLiteApply": AnimaLLLiteApply,

    "AnimaIPAdapterLoader":    AnimaIPAdapterLoader,
    "AnimaIPAdapterApply":     AnimaIPAdapterApply,
    "AnimaSiglipeEncodeImage": AnimaSiglipeEncodeImage,
    "AnimaImageEmbLoader":     AnimaImageEmbLoader,

    "MultiLatentComposite":  MultiLatentComposite,
    "MultiAreaConditioning": MultiAreaConditioning,
    "ConditioningUpscale":   ConditioningUpscale,
    "ConditioningStretch":   ConditioningStretch,

    # Experimental Krea2 nodes
    "ConditioningKrea2Rebalance": ConditioningKrea2Rebalance,

    # Experimental SCAIL2 nodes
    "SCAIL2EasyConfig" : SCAIL2EasyConfig,
    "SCAIL2AutoVideo"  : SCAIL2AutoVideo,
    "SCAIL2RunInfo"    : SCAIL2RunInfo,


    "MiniMaxH3AutoChainMotionContext":     MiniMaxH3AutoChainMotionContext,
    "MiniMaxH3AutoChainMotionContextTrim": MiniMaxH3AutoChainMotionContextTrim,
    "MiniMaxH3AutoChainSaveLatent":        MiniMaxH3AutoChainSaveLatent,
    "MiniMaxH3AutoChainLoadLatent":        MiniMaxH3AutoChainLoadLatent,
    "MiniMaxH3AutoChainAudio":             MiniMaxH3AutoChainAudio,
    "MiniMaxH3AutoChain":                  MiniMaxH3AutoChain,
    "MiniMaxH3AutoChainFrameReference":    MiniMaxH3AutoChainFrameReference,
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
    "FG_SaveVideo"                   : "🗑️ Save Video",
    "FG_SendTelegramNotification"    : "🗑️ Send Telegram Notification",
    "FG_ShowText"                    : "🗑️ Show Text",
    "FG_WD14Tagger"                  : "🗑️ WD14 Tagger (Booru Tags)",
    "FG_MiniMaxH3_Conditioner"       : "🗑️ MiniMax H3 Conditioner",
    "FG_UnifiedModelsLoader"         : "🗑️ Unified Models Loader",

    "FG_XPUGuard"                         : "⚙️ XPU Guard (Device Health)",
    "AnimaConditioningRegion"             : "⚙️ Anima Conditioning Region",
    "ApplyAnimaRegionalConditioningPatch" : "⚙️ Apply Anima Regional Conditioning Patch",
    "AnimaLLLiteApply"                    : "⚙️ Apply Anima ControlNet-LLLite",

    "AnimaIPAdapterLoader"   : "⚙️ Anima IP-Adapter Loader",
    "AnimaIPAdapterApply"    : "⚙️ Anima IP-Adapter Apply",
    "AnimaSiglipeEncodeImage": "⚙️ Anima SigLIP2 Encode Image",
    "AnimaImageEmbLoader"    : "⚙️ Anima Image Embedding Loader (Legacy)",

    "MultiLatentComposite":  "⚙️ Multi Latent Composite",
    "MultiAreaConditioning": "⚙️ Multi Area Conditioning",
    "ConditioningUpscale":   "⚙️ Conditioning Upscale",
    "ConditioningStretch":   "⚙️ Conditioning Stretch",

    "ConditioningKrea2Rebalance": "🎛️ Krea 2 Conditioning Control",

    "SCAIL2EasyConfig": "⚙️ SCAIL-2 Easy Config",
    "SCAIL2AutoVideo":  "⚙️ SCAIL-2 Auto Video (loop)",
    "SCAIL2RunInfo":    "⚙️ SCAIL-2 Run Info",

    "MiniMaxH3AutoChainMotionContext":     "⚙️ H3 Auto Chain Motion Context",
    "MiniMaxH3AutoChainMotionContextTrim": "⚙️ H3 Auto Chain Motion Context Trim",
    "MiniMaxH3AutoChainSaveLatent":        "⚙️ H3 Auto Chain Save Latent",
    "MiniMaxH3AutoChainLoadLatent":        "⚙️ H3 Auto Chain Load Latent",
    "MiniMaxH3AutoChainAudio":             "⚙️ H3 Auto Chain Audio",
    "MiniMaxH3AutoChain":                  "⚙️ H3 Auto Chain + Stitch",
    "MiniMaxH3AutoChainFrameReference":    "⚙️ H3 Auto Chain Frame Reference",
}
