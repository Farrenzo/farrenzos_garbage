import comfy.patcher_extension
from .anima_methods import (
    REGION_TYPE,
    WRAPPER_KEY,
    _prepare_mask,
    _validate_anima_model,
    _diffusion_model_wrapper,
    AnimaConditioningRegionChain,
    AnimaRegionalConditioningPatch
)

# ---------------------------------------------------------------------------
# ComfyUI nodes
# ---------------------------------------------------------------------------

class AnimaConditioningRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "conditioning": ("CONDITIONING",),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "regions": (REGION_TYPE,),
            },
        }

    RETURN_TYPES = (REGION_TYPE,)
    RETURN_NAMES = ("regions",)
    FUNCTION = "create"
    CATEGORY = "Farrenzo's Garbage/Anima/Conditioning"

    def create(self, mask, conditioning, weight, regions=None):
        return (
            AnimaConditioningRegionChain(regions, _prepare_mask(mask), conditioning, float(weight)),
        )


class ApplyAnimaRegionalConditioningPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "regions": (REGION_TYPE,),
                "base_mode": (
                    ["uncovered_only", "global", "disabled"],
                    {"default": "uncovered_only"},
                ),
                "base_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "cross_mask_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "self_mask_strength": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "base_ratio": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "cross_inject_every_n_blocks": (
                    "INT",
                    {"default": 1, "min": 1, "max": 100, "step": 1},
                ),
                "self_inject_every_n_blocks": (
                    "INT",
                    {"default": 1, "min": 1, "max": 100, "step": 1},
                ),
            },
            "optional": {
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "background_conditioning": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("patched_model",)
    FUNCTION = "apply"
    CATEGORY = "Farrenzo's Garbage/Anima/Conditioning"

    def apply(
        self,
        model,
        regions,
        base_mode,
        base_strength,
        end_percent,
        cross_mask_strength,
        self_mask_strength,
        base_ratio,
        cross_inject_every_n_blocks,
        self_inject_every_n_blocks,
        start_percent=0.0,
        background_conditioning=None,
    ):
        _validate_anima_model(model)
        model_sampling = model.get_model_object("model_sampling")
        start_sigma = float(model_sampling.percent_to_sigma(start_percent))
        end_sigma = float(model_sampling.percent_to_sigma(end_percent))
        patch = AnimaRegionalConditioningPatch(
            regions.flatten(),
            base_mode,
            base_strength,
            start_sigma,
            end_sigma,
            cross_mask_strength,
            self_mask_strength,
            base_ratio,
            cross_inject_every_n_blocks,
            self_inject_every_n_blocks,
            background_conditioning,
        )

        patched_model = model.clone()
        patched_model.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY
        )
        patched_model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            _diffusion_model_wrapper,
        )
        patched_model.model_options.setdefault("transformer_options", {})[WRAPPER_KEY] = patch
        patched_model.set_attachments(WRAPPER_KEY, patch)
        return (patched_model,)


