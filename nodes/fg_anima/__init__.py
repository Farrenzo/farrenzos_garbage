from .fg_anima_controlnet      import FG_AnimaLLLiteApply
from .fg_anima_role_ref_kv     import FG_AnimaRoleRefKV

ANIMA_NODE_CLASS_MAPPINGS = {
    "FG_AnimaRoleRefKV":FG_AnimaRoleRefKV,
    "FG_AnimaLLLiteApply":FG_AnimaLLLiteApply,
}

ANIMA_DISPLAY_NAME_MAPPINGS = {
    "FG_AnimaRoleRefKV"  : "🗑️ Anima Role Reference KV Patch",
    # Experimental Anima nodes
    "FG_AnimaLLLiteApply": "⚙️ Anima Apply ControlNet-LLLite",
}

from .fg_anima_ipadapter_nodes import (
    ANIMA_IP_ADAPTER_NODE_CLASS_MAPPINGS,
    ANIMA_IP_ADAPTER_DISPLAY_NAME_MAPPINGS
)

from .fg_anima_model_patch_nodes import (
    ANIMA_PATCH_NODE_CLASS_MAPPINGS,
    ANIMA_PATCH_NODE_DISPLAY_NAME_MAPPINGS
)

ANIMA_NODE_CLASS_MAPPINGS.update(ANIMA_IP_ADAPTER_NODE_CLASS_MAPPINGS)
ANIMA_DISPLAY_NAME_MAPPINGS.update(ANIMA_IP_ADAPTER_DISPLAY_NAME_MAPPINGS)

ANIMA_NODE_CLASS_MAPPINGS.update(ANIMA_PATCH_NODE_CLASS_MAPPINGS)
ANIMA_DISPLAY_NAME_MAPPINGS.update(ANIMA_PATCH_NODE_DISPLAY_NAME_MAPPINGS)
