from ._fg_helperfunctions import log
from .fg_intelCode import install_xpu_patch, start_keepalive

from .fg_advanced_ksampler    import FG_Advanced_KSampler
from .fg_canny_detect         import FG_CannyDetect
from .fg_CLIP_text_encode     import FG_CLIPTextEncode
from .fg_controlnet           import FG_ApplyControlNet
from .fg_coordinates_box_fill import FG_CoordinatesBoxFill, FG_MaskBackdrop
from .fg_empty_latent         import FG_EmptyLatent
from .fg_image_scale          import FG_ImageScaler, FG_CropFactorPadding
from .fg_lab_color_transfer   import FG_LABColorTransfer
from .fg_load_vae             import FG_LatentTransfer
from .fg_load_image           import FG_LoadImage
from .fg_lora_loader          import FG_LoraLoader
from .fg_math_utils           import FG_MinimumMaximum, FG_IntFloatOut
from .fg_model_reference      import FG_ModelReferenceLatentMethod
from .fg_resource_manager     import FG_ResourceManager
from .fg_purge_memory         import FG_PurgeMemory
from .fg_save_image           import FG_SaveImage
from .fg_save_video           import FG_SaveVideo
from .fg_show_text            import FG_ShowText
from .fg_telegram_notice      import FG_SendTelegramNotification
from .fg_WD14                 import FG_WD14Tagger
from .fg_xpu_guard            import FG_XPUGuard
from .fg_unified_loader       import FG_UnifiedModelsLoader
from .fg_hue_correct          import FG_HueCorrect
from .fg_image_forensics      import FG_ImageForensics
from .fg_conditioning_cache   import FG_SaveConditioningLatent, FG_LoadConditioningLatent
from .fg_step_inspect         import FG_StepInspect
from .fg_noisy_seeder         import FG_NoisySeeder
from .fg_seed_variance_enhancer import FG_SeedVarianceEnhancer
from .fg_linework_composite import (
    FG_ColorMap,
    FG_BubbleTint,
    FG_ColorPicker,
    FG_LineworkComposite
)

from .fg_anima.anima_lllite_inspect     import FG_LLLiteCondPreview
from .fg_anima.anima_controlnet_nodes   import FG_AnimaLLLiteApply
from .fg_anima.fg_easy_controller_patch import FG_AnimaEasyControlPatch
from .fg_anima.anima_regional_prompt_nodes import ApplyAnimaRegionalConditioningPatch, FG_MaskFromRGBCMYBW
from .fg_anima.anima_ipadapter_nodes import (
    AnimaIPAdapterLoader,
    AnimaIPAdapterApply,
    AnimaSiglipeEncodeImage,
    AnimaIPAdapterVisualize
)
from .fg_anima.anima_cosmos_latent import FG_ApplyCosmosReferenceLatent

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
from .fg_yue2.fg_yue2_node import FG_YuE2XPUAccel, FG_MiniMaxMusicXPUAccel

from .fg_inpaint_canvas import (
    NODE_CLASS_MAPPINGS as _INPAINT_CANVAS_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _INPAINT_CANVAS_NAMES,
)  # Inpaint Canvas (ported from DenRakEiw/ComfyUI-InpaintCanvas, GPL-3.0)

NODE_CLASS_MAPPINGS = {
    "FG_Advanced_KSampler"           : FG_Advanced_KSampler,
    "FG_ApplyControlNet"             : FG_ApplyControlNet,
    "FG_ApplyCosmosReferenceLatent"  : FG_ApplyCosmosReferenceLatent,
    "FG_BoxFillwCoordinates"         : FG_CoordinatesBoxFill,
    "FG_CropFactorPadding"           : FG_CropFactorPadding,
    "FG_BubbleTint"                  : FG_BubbleTint,
    "FG_CannyDetect"                 : FG_CannyDetect,
    "FG_CLIPTextEncode"              : FG_CLIPTextEncode,
    "FG_DynamicLoraLoader"           : FG_LoraLoader,
    "FG_EmptyLatent"                 : FG_EmptyLatent,
    "FG_ImageScaler"                 : FG_ImageScaler,
    "FG_IntFloatOut"                 : FG_IntFloatOut,
    "FG_LABColorTransfer"            : FG_LABColorTransfer,
    "FG_LatentTransfer"              : FG_LatentTransfer,
    "FG_LineworkComposite"           : FG_LineworkComposite,
    "FG_LoadConditioning"            : FG_LoadConditioningLatent,
    "FG_LoadImage"                   : FG_LoadImage,
    "FG_MiniMaxH3_Conditioner"       : FG_MiniMaxH3_Conditioner,
    "FG_Minimum_Maximum"             : FG_MinimumMaximum,
    "FG_ModelReferenceLatentMethod"  : FG_ModelReferenceLatentMethod,
    "FG_NoisySeeder"                 : FG_NoisySeeder,
    "FG_PurgeMemory"                 : FG_PurgeMemory,
    "FG_ResourceManager"             : FG_ResourceManager,
    "FG_SaveConditioning"            : FG_SaveConditioningLatent,
    "FG_SaveImage"                   : FG_SaveImage,
    "FG_SaveVideo"                   : FG_SaveVideo,
    "FG_SendTelegramNotification"    : FG_SendTelegramNotification,
    "FG_ShowText"                    : FG_ShowText,
    "FG_UnifiedModelsLoader"         : FG_UnifiedModelsLoader,
    "FG_WD14Tagger"                  : FG_WD14Tagger,
    "FG_XPUGuard"                    : FG_XPUGuard,
    "FG_YuE2XPUAccel"                : FG_YuE2XPUAccel,
    "FG_MiniMaxMusicXPUAccel"        : FG_MiniMaxMusicXPUAccel,

    # Experimental Anima nodes
    "FG_AnimaLLLiteApply"                : FG_AnimaLLLiteApply,
    "FG_AnimaEasyControlPatch"           : FG_AnimaEasyControlPatch,
    "FG_LLLiteCondPreview"               : FG_LLLiteCondPreview,
    "ApplyAnimaRegionalConditioningPatch": ApplyAnimaRegionalConditioningPatch,
    "FG_MaskFromRGBCMYBW+": FG_MaskFromRGBCMYBW,

    "AnimaIPAdapterLoader":    AnimaIPAdapterLoader,
    "AnimaIPAdapterApply":     AnimaIPAdapterApply,
    "AnimaSiglipeEncodeImage": AnimaSiglipeEncodeImage,
    "AnimaIPAdapterVisualize": AnimaIPAdapterVisualize,

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

    # Experimental
    "FG_HueCorrect"          : FG_HueCorrect,
    "FG_ImageForensics"      : FG_ImageForensics,
    "FG_StepInspect"         : FG_StepInspect,
    "FG_MaskBackdrop"        : FG_MaskBackdrop,
    "FG_ColorPicker"         : FG_ColorPicker,
    "FG_ColorMap"            : FG_ColorMap,
    "FG_SeedVarianceEnhancer": FG_SeedVarianceEnhancer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FG_Advanced_KSampler"           : "🗑️ Advanced KSampler",
    "FG_ApplyControlNet"             : "🗑️ Apply Advanced ControlNet",
    "FG_ApplyCosmosReferenceLatent"  : "🗑️ Anima Apply Cosmos Reference Latent",
    "FG_BoxFillwCoordinates"         : "🗑️ Coordinate Box Fill",
    "FG_BubbleTint"                  : "🗑️ Bubble Tint (colour lettering)",
    "FG_CannyDetect"                 : "🗑️ Advanced Edge Detector",
    "FG_CLIPTextEncode"              : "🗑️ Enhanced CLIP Text Encode",
    "FG_CombinedImageTagger"         : "🗑️ Combined Image Tagger",
    "FG_DynamicLoraLoader"           : "🗑️ Multi-LoRA Loader",
    "FG_EmptyLatent"                 : "🗑️ Advanced Empty Latent",
    "FG_ImageScaler"                 : "🗑️ Image Scaler",
    "FG_IntFloatOut"                 : "🗑️ Int / Float Out",
    "FG_CropFactorPadding"           : "🗑️ Crop Factor Padding",
    "FG_KSampler"                    : "🗑️ KSampler for Qwen Image Edit",
    "FG_LABColorTransfer"            : "🗑️ LAB Color Transfer",
    "FG_LatentTransfer"              : "🗑️ Latent Transfer (VAE to VAE)",
    "FG_LineworkComposite"           : "🗑️ Linework Composite",
    "FG_LoadConditioning"            : "🗑️ Load Conditioning From Disk",
    "FG_LoadImage"                   : "🗑️ Load Image",
    "FG_MiniMaxH3_Conditioner"       : "🗑️ MiniMax H3 Conditioner",
    "FG_Minimum_Maximum"             : "🗑️ Minimum + Maximum",
    "FG_ModelReferenceLatentMethod"  : "🗑️ Edit Model Reference Method",
    "FG_PurgeMemory"                 : "🗑️ Purge Memory",
    "FG_SaveConditioning"            : "🗑️ Save Conditioning To Disk",
    "FG_SaveImage"                   : "🗑️ Save Image",
    "FG_SaveVideo"                   : "🗑️ Save Video",
    "FG_SendTelegramNotification"    : "🗑️ Send Telegram Notification",
    "FG_ShowText"                    : "🗑️ Show Text",
    "FG_UnifiedModelsLoader"         : "🗑️ Unified Models Loader",
    "FG_WD14Tagger"                  : "🗑️ WD14 Tagger (Booru Tags)",
    "FG_NoisySeeder"                 : "🗑️ Noisy Seeder",

    "FG_ResourceManager"             : "🗑️ Resource Manager",
    "FG_XPUGuard"                    : "🗑️ XPU Guard (Device Health)",
    "FG_YuE2XPUAccel"                : "🗑️ YuE2 XPU Accelerator",
    "FG_MiniMaxMusicXPUAccel"        : "🗑️ MiniMaxMusicH3 XPU Accelerator",

    "AnimaIPAdapterApply"    : "🗑️ Anima IP-Adapter Apply",
    "AnimaIPAdapterLoader"   : "🗑️ Anima IP-Adapter Loader",
    "AnimaIPAdapterVisualize": "🗑️ Anima IP Adapter Visualizer",
    "AnimaSiglipeEncodeImage": "🗑️ Anima SigLIP2 Encode Image",

    "FG_AnimaLLLiteApply"                : "🗑️ Anima Apply ControlNet-LLLite",
    "FG_AnimaEasyControlPatch"           : "🗑️ Anima EasyControl (KSampler)",
    "FG_LLLiteCondPreview"               : "⚙️ LLLite Cond Preview",
    "ApplyAnimaRegionalConditioningPatch": "⚙️ Anima Regional Conditioning",
    "FG_MaskFromRGBCMYBW+": "🔧 Mask From RGB/CMY/BW",

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

    "UnetLoaderGGUF"          : "🎛️ UnetLoaderGGUF",
    "CLIPLoaderGGUF"          : "🎛️ CLIPLoaderGGUF",
    "DualCLIPLoaderGGUF"      : "🎛️ DualCLIPLoaderGGUF",
    "TripleCLIPLoaderGGUF"    : "🎛️ TripleCLIPLoaderGGUF",
    "QuadrupleCLIPLoaderGGUF" : "🎛️ QuadrupleCLIPLoaderGGUF",
    "UnetLoaderGGUFAdvanced"  : "🎛️ UnetLoaderGGUFAdvanced",

    # Experimental
    "FG_HueCorrect"    : "🗑️ Hue Correct (LAB)",
    "FG_ImageForensics": "🗑️ Image Forensics",
    "FG_StepInspect"   : "🗑️ Step Inspect (x0 trajectory)",
    "FG_MaskBackdrop"  : "🗑️ Mask Backdrop",

    "FG_ColorPicker": "🗑️ Color Picker",
    "FG_ColorMap": "🗑️ Color Map (for Bubble Tint)",
    "FG_SeedVarianceEnhancer": "🎲 Seed Variance Enhancer"
}

# Inpaint Canvas: keep the upstream class keys so Scumble / Renzo Studio recipes match.
NODE_CLASS_MAPPINGS.update(_INPAINT_CANVAS_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update({k: "🎨 " + v for k, v in _INPAINT_CANVAS_NAMES.items()})

