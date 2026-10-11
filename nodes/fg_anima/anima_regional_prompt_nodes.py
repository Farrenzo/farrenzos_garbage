import json

import torch
import comfy.patcher_extension
from .anima_methods import (
    WRAPPER_KEY,
    _validate_anima_model,
    _diffusion_model_wrapper,
    AnimaConditioningRegion,
    AnimaRegionalConditioningPatch
)

# ---------------------------------------------------------------------------
# ComfyUI nodes
# ---------------------------------------------------------------------------

DEFAULT_JSON_PROMPT = json.dumps({
    "width": 1216,
    "height": 832,
    "base_prompt": "",
    "negative_prompt": "",
    "regions": [
        {"name": "Region 1", "prompt": "", "x": 0,   "y": 0, "width": 608, "height": 832, "strength": 0.55},
        {"name": "Region 2", "prompt": "", "x": 608, "y": 0, "width": 608, "height": 832, "strength": 0.85},
    ],
}, indent=2)


class ApplyAnimaRegionalConditioningPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "base_mode": (
                    ["uncovered_only", "global", "disabled"],
                    {"default": "uncovered_only"},
                ),
                "base_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
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
                "json_prompt": ("STRING", {"multiline": True, "default": DEFAULT_JSON_PROMPT}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("patched_model", "positive", "negative",)
    FUNCTION = "apply"
    CATEGORY = "Farrenzo's Garbage/Anima/Conditioning"

    def apply(
        self,
        model,
        clip,
        base_mode,
        base_strength,
        start_percent,
        end_percent,
        cross_mask_strength,
        self_mask_strength,
        base_ratio,
        cross_inject_every_n_blocks,
        self_inject_every_n_blocks,
        json_prompt,
    ):
        _validate_anima_model(model)
        data = json.loads(json_prompt)
        canvas_w = int(data["width"])
        canvas_h = int(data["height"])

        def encode(prompt):
            return clip.encode_from_tokens_scheduled(clip.tokenize(prompt))

        regions = []
        for region in data["regions"]:
            x, y = int(region["x"]), int(region["y"])
            mask = torch.zeros(1, canvas_h, canvas_w)
            mask[:, max(y, 0):y + int(region["height"]), max(x, 0):x + int(region["width"])] = 1.0
            regions.append(AnimaConditioningRegion(mask, encode(region["prompt"]), float(region["strength"])))
        if not regions:
            raise RuntimeError("Anima Regional Conditioning: JSON prompt has no regions.")

        base_prompt = data.get("base_prompt", "")
        positive = encode(base_prompt)
        negative = encode(data.get("negative_prompt", ""))

        model_sampling = model.get_model_object("model_sampling")
        start_sigma = float(model_sampling.percent_to_sigma(start_percent))
        end_sigma = float(model_sampling.percent_to_sigma(end_percent))
        patch = AnimaRegionalConditioningPatch(
            regions,
            base_mode,
            base_strength,
            start_sigma,
            end_sigma,
            cross_mask_strength,
            self_mask_strength,
            base_ratio,
            cross_inject_every_n_blocks,
            self_inject_every_n_blocks,
            positive if base_prompt.strip() else None,
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
        return (patched_model, positive, negative,)


class FG_MaskFromRGBCMYBW:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "threshold_r": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
                "threshold_g": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
                "threshold_b": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
            }
        }

    RETURN_TYPES = ("MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK",)
    RETURN_NAMES = ("red","green","blue","cyan","magenta","yellow","black","white",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, image, threshold_r, threshold_g, threshold_b):
        red = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] < threshold_b)).float()
        green = ((image[..., 0] < threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] < threshold_b)).float()
        blue = ((image[..., 0] < threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] >= 1-threshold_b)).float()

        cyan = ((image[..., 0] < threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
        magenta = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] > 1-threshold_b)).float()
        yellow = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] < threshold_b)).float()

        black = ((image[..., 0] <= threshold_r) & (image[..., 1] <= threshold_g) & (image[..., 2] <= threshold_b)).float()
        white = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
        
        return (red, green, blue, cyan, magenta, yellow, black, white,)


