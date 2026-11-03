from ._fg_helperfunctions import log
from .fg_intelCode import install_xpu_patch, start_keepalive

from .fg_advanced_ksampler    import FG_Advanced_KSampler
from .fg_canny_detect         import FG_CannyDetect
from .fg_CLIP_text_encode     import FG_CLIPTextEncode
from .fg_controlnet           import FG_ApplyControlNet
from .fg_empty_latent         import FG_EmptyLatent
from .fg_image_scale          import FG_ImageScaler, FG_CropFactorPadding
from .fg_lab_color_transfer   import FG_LABColorTransfer
from .fg_load_vae             import FG_LatentTransfer
from .fg_load_image           import FG_LoadImage
from .fg_lora_loader          import FG_LoraLoader
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

from .fg_minimax.fg_minimax_h3 import FG_MiniMaxH3_Conditioner
from .fg_yue2.fg_yue2_node import FG_YuE2XPUAccel, FG_MiniMaxMusicXPUAccel

from .fg_anima.anima_lllite_inspect     import FG_LLLiteCondPreview
from .fg_anima.anima_controlnet_nodes   import FG_AnimaLLLiteApply
from .fg_anima.fg_easy_controller_patch import FG_AnimaEasyControlPatch
from .fg_anima.fg_anima_role_ref_kv     import FG_AnimaRoleRefKV
from .fg_anima.anima_regional_prompt_nodes import ApplyAnimaRegionalConditioningPatch, FG_MaskFromRGBCMYBW

from .fg_anima.anima_cosmos_latent import FG_ApplyCosmosReferenceLatent

NODE_CLASS_MAPPINGS = {
    "FG_Advanced_KSampler"           : FG_Advanced_KSampler,
    "FG_ApplyControlNet"             : FG_ApplyControlNet,
    "FG_ApplyCosmosReferenceLatent"  : FG_ApplyCosmosReferenceLatent,
    "FG_CropFactorPadding"           : FG_CropFactorPadding,
    "FG_CannyDetect"                 : FG_CannyDetect,
    "FG_CLIPTextEncode"              : FG_CLIPTextEncode,
    "FG_DynamicLoraLoader"           : FG_LoraLoader,
    "FG_EmptyLatent"                 : FG_EmptyLatent,
    "FG_ImageScaler"                 : FG_ImageScaler,
    "FG_LABColorTransfer"            : FG_LABColorTransfer,
    "FG_LatentTransfer"              : FG_LatentTransfer,
    "FG_LoadConditioning"            : FG_LoadConditioningLatent,
    "FG_LoadImage"                   : FG_LoadImage,
    "FG_MiniMaxH3_Conditioner"       : FG_MiniMaxH3_Conditioner,
    "FG_ModelReferenceLatentMethod"  : FG_ModelReferenceLatentMethod,
    "FG_PurgeMemory"                 : FG_PurgeMemory,
    "FG_ResourceManager"             : FG_ResourceManager,
    "FG_AnimaRoleRefKV"              : FG_AnimaRoleRefKV,
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

    # Experimental
    "FG_HueCorrect"          : FG_HueCorrect,
    "FG_ImageForensics"      : FG_ImageForensics,
    "FG_StepInspect"         : FG_StepInspect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FG_Advanced_KSampler"           : "🗑️ Advanced KSampler",
    "FG_ApplyControlNet"             : "🗑️ Apply Advanced ControlNet",
    "FG_ApplyCosmosReferenceLatent"  : "🗑️ Anima Apply Cosmos Reference Latent",
    "FG_CannyDetect"                 : "🗑️ Advanced Edge Detector",
    "FG_CLIPTextEncode"              : "🗑️ Enhanced CLIP Text Encode",
    "FG_CombinedImageTagger"         : "🗑️ Combined Image Tagger",
    "FG_DynamicLoraLoader"           : "🗑️ Multi-LoRA Loader",
    "FG_EmptyLatent"                 : "🗑️ Advanced Empty Latent",
    "FG_ImageScaler"                 : "🗑️ Image Scaler",
    "FG_CropFactorPadding"           : "🗑️ Crop Factor Padding",
    "FG_KSampler"                    : "🗑️ KSampler for Qwen Image Edit",
    "FG_LABColorTransfer"            : "🗑️ LAB Color Transfer",
    "FG_LatentTransfer"              : "🗑️ Latent Transfer (VAE to VAE)",
    "FG_LoadConditioning"            : "🗑️ Load Conditioning From Disk",
    "FG_LoadImage"                   : "🗑️ Load Image",
    "FG_MiniMaxH3_Conditioner"       : "🗑️ MiniMax H3 Conditioner",
    "FG_ModelReferenceLatentMethod"  : "🗑️ Edit Model Reference Method",
    "FG_PurgeMemory"                 : "🗑️ Purge Memory",
    "FG_AnimaRoleRefKV"              : "🗑️ Anima Role Reference KV Patch",
    "FG_SaveConditioning"            : "🗑️ Save Conditioning To Disk",
    "FG_SaveImage"                   : "🗑️ Save Image",
    "FG_SaveVideo"                   : "🗑️ Save Video",
    "FG_SendTelegramNotification"    : "🗑️ Send Telegram Notification",
    "FG_ShowText"                    : "🗑️ Show Text",
    "FG_UnifiedModelsLoader"         : "🗑️ Unified Models Loader",
    "FG_WD14Tagger"                  : "🗑️ WD14 Tagger (Booru Tags)",
    "FG_ResourceManager"             : "🗑️ Resource Manager",
    "FG_XPUGuard"                    : "🗑️ XPU Guard (Device Health)",
    "FG_YuE2XPUAccel"                : "🗑️ YuE2 XPU Accelerator",
    "FG_MiniMaxMusicXPUAccel"        : "🗑️ MiniMaxMusicH3 XPU Accelerator",

    "FG_AnimaLLLiteApply"                : "🗑️ Anima Apply ControlNet-LLLite",
    "FG_AnimaEasyControlPatch"           : "🗑️ Anima EasyControl (KSampler)",
    "FG_LLLiteCondPreview"               : "⚙️ LLLite Cond Preview",
    "ApplyAnimaRegionalConditioningPatch": "⚙️ Anima Regional Conditioning",
    "FG_MaskFromRGBCMYBW+": "🔧 Mask From RGB/CMY/BW",

    # Experimental
    "FG_HueCorrect"    : "🗑️ Hue Correct (LAB)",
    "FG_ImageForensics": "🗑️ Image Forensics",
    "FG_StepInspect"   : "🗑️ Step Inspect (x0 trajectory)",
}

from .fg_anima.anima_ipadapter_nodes import (
    ANIMA_IP_ADAPTER_NODE_CLASS_MAPPINGS,
    ANIMA_IP_ADAPTER_DISPLAY_NAME_MAPPINGS
)

from .fg_krea2_rebalance import (
    KREA2_NODE_CLASS_MAPPINGS,
    KREA2_NODE_DISPLAY_NAME_MAPPINGS
)

from .fg_image_mask_utils import (
    MASK_UTILS_NODE_CLASS_MAPPINGS,
    MASK_UTILS_NODE_DISPLAY_NAME_MAPPINGS
)

from .fg_seed_utils import (
    SEED_UTILS_NODE_CLASS_MAPPINGS,
    SEED_UTILS_DISPLAY_NAME_MAPPINGS,
)

from .fg_linework_composite import (
    LINEWORK_NODE_CLASS_MAPPINGS,
    LINEWORK_DISPLAY_NAME_MAPPINGS,
)

from .fg_minimax.fg_minimax_autochain import (
    AUTOCHAIN_NODE_CLASS_MAPPINGS,
    AUTOCHAIN_DISPLAY_NAME_MAPPINGS
)

from .fg_regional_prompt_nodes import (
    REGIONAL_PROMPT_NODE_CLASS_MAPPINGS,
    REGIONAL_PROMPT_DISPLAY_NAME_MAPPINGS,
)

from .fg_SCAIL2 import (
    SCAIL2_NODE_CLASS_MAPPINGS,
    SCAIL2_DISPLAY_NAME_MAPPINGS
)

from .fg_math_utils import (
    MATH_UTILS_NODE_CLASS_MAPPINGS,
    MATH_UTILS_NODE_DISPLAY_NAME_MAPPINGS,
)

from .fg_inpaint_canvas import (
    INPAINT_CANVAS_NODE_CLASS_MAPPINGS,
    INPAINT_CANVAS_NODE_DISPLAY_NAME_MAPPINGS,
)  # Inpaint Canvas (ported from DenRakEiw/ComfyUI-InpaintCanvas, GPL-3.0)

NODE_CLASS_MAPPINGS.update(ANIMA_IP_ADAPTER_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ANIMA_IP_ADAPTER_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(KREA2_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(KREA2_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(MASK_UTILS_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MASK_UTILS_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(SEED_UTILS_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SEED_UTILS_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(LINEWORK_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(LINEWORK_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(AUTOCHAIN_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(AUTOCHAIN_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(REGIONAL_PROMPT_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(REGIONAL_PROMPT_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(SCAIL2_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SCAIL2_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(MATH_UTILS_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MATH_UTILS_NODE_DISPLAY_NAME_MAPPINGS)

# Inpaint Canvas: keep the upstream class keys so Scumble / Renzo Studio recipes match.
NODE_CLASS_MAPPINGS.update(INPAINT_CANVAS_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(INPAINT_CANVAS_NODE_DISPLAY_NAME_MAPPINGS)
