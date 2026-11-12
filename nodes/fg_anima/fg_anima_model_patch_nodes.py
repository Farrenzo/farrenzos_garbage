import json
import torch
import torch.nn.functional as F
from typing import Any

from .anima_methods import (
    WRAPPER_KEY,
    _validate_anima_model,
    _diffusion_model_wrapper,
    cosmos_extra_conds_reference,
    cosmos_diffusion_reference_wrapper,
    AnimaConditioningRegion,
    AnimaRegionalConditioningPatch
)

import folder_paths
import comfy.patcher_extension
from comfy.model_base import Anima
from comfy.model_patcher import ModelPatcher
from comfy_api.latest import io

from .fg_anima_controlnet import (
    _build_inpaint_cond_image,
    _prepare_cond_image,
    _prepare_mask,
    _target_cond_hw,
    read_lllite_metadata,
)


MAX_REF_LATENTS = 3
TEMPORAL_REFERENCE_WRAPPER_KEY = "cosmos_temporal_reference"

class FG_ApplyCosmosReferenceLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FG_ApplyCosmosReferenceLatent",
            search_aliases=["cosmos reference", "anima reference"],
            display_name="Apply Cosmos Reference Latent",
            category="Farrenzo's Garbage/Anima/Conditioning",
            inputs=[
                io.Model.Input("model"),
                io.Autogrow.Input(
                    "ref_latents",
                    template=io.Autogrow.TemplateNames(
                        io.Latent.Input("ref_latent"),
                        names=[f"ref_latent_{i}" for i in range(1, MAX_REF_LATENTS + 1)],
                        min=0,
                    ),
                    tooltip=f"Reference latent(s) for generation. Up to {MAX_REF_LATENTS} latents.",
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        ref_latents: dict[str, dict[str, Any]] = kwargs["ref_latents"]
        if 'latent' in kwargs:
            ref_latents['latent_for_compatibility'] = kwargs["latent"]
        m = model.clone()
        model_type = type(m.model)

        if issubclass(model_type, Anima):
            extra_conds = m.get_model_object("extra_conds")
            process_latent_in = m.get_model_object("process_latent_in")
            m.add_object_patch(
                "extra_conds",
                cosmos_extra_conds_reference(extra_conds, process_latent_in, ref_latents),
            )
            m.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                TEMPORAL_REFERENCE_WRAPPER_KEY,
                cosmos_diffusion_reference_wrapper,
            )

        return io.NodeOutput(m)


"""
FG_LLLiteInspect — see what the ControlNet is actually handed, before you sample.

Two tools:

  FG_LLLiteCondPreview   Static. Runs the REAL preprocessing functions from
                         anima_controlnet_nodes (imported, not reimplemented, so
                         this can never drift from the live path) and shows you
                         the exact 4-channel tensor LLLite will receive, decoded
                         back into viewable images. Plus the token arithmetic
                         that decides whether LLLite fires at all.

  install_lllite_counters()  Runtime. Wraps LLLiteModuleDiT.forward to count how
                         many modules FIRED versus BYPASSED, and how large the
                         injected delta was relative to the activations it was
                         added to. Prints one summary line per sampling step.
                         Call it once from your pack's __init__.

The single most common LLLite failure is silent: when the DiT's token count does
not equal the cond image's token count, every module returns org_forward with no
error. The preview reports the numbers; the counters prove it at runtime.
"""

# _Conditioning1: Conv(k4,s4) -> Conv(k3,s1) -> Conv(k4,s4) = /16 overall
_COND_STRIDE = 16
FOLDER = "model_patches"


def _to_image(t):
    """(1,C,H,W) in [-1,1] -> ComfyUI IMAGE (1,H,W,3) in [0,1]."""
    x = ((t.float() + 1.0) * 0.5).clamp(0, 1)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x.permute(0, 2, 3, 1)


class FG_LLLiteCondPreview:
    """Show the exact conditioning tensor an LLLite checkpoint will receive."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lllite_name": (folder_paths.get_filename_list(FOLDER),),
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
                "latent": ("LATENT", {
                    "tooltip": "Wire the SAME latent going to your sampler. Without "
                               "it the preview assumes image size / 8, which is only "
                               "right if you aren't rescaling anywhere."}),
                "model": ("MODEL", {"tooltip": "Optional, only to read patch_spatial."}),
                "reference_frames": ("INT", {
                    "default": 1, "min": 1, "max": 8,
                    "tooltip": "Set to 2 if ApplyCosmosReferenceLatent is in the "
                               "graph with one reference. This is what decides "
                               "whether LLLite fires."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("cond_rgb", "cond_mask", "overlay", "report")
    FUNCTION = "preview"
    CATEGORY = "Farrenzo's Garbage/Controlnet/Anima"
    DESCRIPTION = "Decode the LLLite cond image back to something you can look at."

    def preview(self, lllite_name, image, mask=None, latent=None, model=None,
                reference_frames=1):
        path = folder_paths.get_full_path(FOLDER, lllite_name)
        meta = read_lllite_metadata(path)
        cond_in_channels = int(meta.get("lllite.cond_in_channels", 3))
        masked_input = str(meta.get("lllite.inpaint_masked_input", "false")).lower() == "true"

        patch = 2
        if model is not None:
            try:
                patch = int(getattr(model.get_model_object("diffusion_model"), "patch_spatial", 2))
            except Exception:
                pass

        if latent is not None:
            lat = latent["samples"]
            latent_h, latent_w = int(lat.shape[-2]), int(lat.shape[-1])
            src = "latent input"
        else:
            latent_h, latent_w = image.shape[1] // 8, image.shape[2] // 8
            src = "image size / 8 (assumed)"

        dev, dt = torch.device("cpu"), torch.float32
        rgb = _prepare_cond_image(image, latent_h, latent_w, dev, dt, patch)

        lines = [
            f"checkpoint      : {lllite_name}",
            f"cond_in_channels: {cond_in_channels}"
            + ("  (INPAINT — mask is used)" if cond_in_channels == 4
               else "  (mask input is IGNORED by this checkpoint)"),
            f"masked_input    : {masked_input}"
            + ("  -> RGB is ZEROED wherever mask >= 0.5" if masked_input else ""),
            f"target_atomics  : {meta.get('lllite.target_atomics', meta.get('lllite.target_layers', '?'))}",
            f"latent dims     : {latent_h} x {latent_w}   ({src})",
            f"cond image      : {tuple(rgb.shape)}  "
            f"(target {_target_cond_hw(latent_h, latent_w, patch)})",
        ]

        if cond_in_channels == 4:
            if mask is None:
                raise ValueError(
                    f"'{lllite_name}' is a 4-channel inpaint checkpoint and needs a "
                    "MASK to preview. Connect the same mask your apply node gets."
                )
            mk = _prepare_mask(mask, latent_h, latent_w, dev, dt, patch)
            cond = _build_inpaint_cond_image(rgb, mk, masked_input)
            cov = float(mk.mean()) * 100
            rgb_out = cond[:, :3]
            zeroed = float((rgb_out.abs().amax(dim=1, keepdim=True) < 1e-6).float().mean()) * 100
            lines += [
                f"mask coverage   : {cov:.1f}% of the frame is inpaint area",
                f"RGB blacked out : {zeroed:.1f}% of pixels carry no image information",
            ]
            if masked_input and cov > 60:
                lines.append(
                    "  !! Over 60% of the cond image is masked and zeroed. The model "
                    "has almost no context. Crop with 2-3x padding around the mask "
                    "bbox instead of tight to it."
                )
            mask_view = _to_image(cond[:, 3:4])
        else:
            cond = rgb
            mask_view = torch.zeros(1, rgb.shape[2], rgb.shape[3], 3)
            if mask is not None:
                lines.append(
                    "  !! You connected a MASK but this checkpoint is 3-channel. "
                    "The apply node drops it with only a warning."
                )

        # --- the arithmetic that decides whether LLLite runs at all
        pad_h = ((latent_h + patch - 1) // patch) * patch
        pad_w = ((latent_w + patch - 1) // patch) * patch
        cond_tokens = (pad_h * 8 // _COND_STRIDE) * (pad_w * 8 // _COND_STRIDE)
        dit_tokens = cond_tokens * reference_frames

        lines += [
            "",
            f"cond tokens     : {cond_tokens:,}",
            f"DiT tokens      : {dit_tokens:,}   ({reference_frames} frame"
            f"{'s' if reference_frames != 1 else ''})",
        ]
        if cond_tokens == dit_tokens:
            lines.append("MATCH -> every LLLite module will fire.")
        else:
            lines.append(
                f"MISMATCH -> LLLiteModuleDiT.forward returns org_forward for EVERY "
                f"module. LLLite contributes exactly nothing, with no error printed. "
                f"Cause: {reference_frames} frames in the sequence but the cond image "
                f"only covers 1. Apply the token-mask patch, or remove "
                f"ApplyCosmosReferenceLatent."
            )

        rgb_view = _to_image(cond[:, :3])
        if cond_in_channels == 4:
            m = ((cond[:, 3:4].float() + 1) * 0.5).clamp(0, 1).permute(0, 2, 3, 1)
            tint = torch.tensor([1.0, 0.15, 0.15]).view(1, 1, 1, 3)
            overlay = (rgb_view * (1 - m * 0.55) + tint * m * 0.55).clamp(0, 1)
        else:
            overlay = rgb_view

        report = "\n".join(lines)
        print("[FG_LLLiteCondPreview]\n" + report)
        return (rgb_view, mask_view, overlay, report)


"""
REGIONAL PROMPT
"""

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


class FG_AnimaRegionalPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "base_mode": (
                    ["uncovered_only", "global", "disabled"],
                    {"default": "uncovered_only", "tooltip": (
                        "Where the base_prompt applies.\n"
                        "uncovered_only: only the area outside every region box.\n"
                        "global: the whole image, layered under each region's own prompt. "
                        "This also removes self-attention separation between regions, so self_mask_strength has no effect.\n"
                        "disabled: nowhere, except as a fallback for pixels outside every box."
                    )},
                ),
                "base_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": (
                        "How strictly the base area is kept to the base prompt, like a region's own strength. "
                        "0 removes the base area. Anything above 1.0 acts the same as 1.0."
                    )},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": (
                        "When regional prompting starts during sampling. 0.0 = the first (noisiest) step. "
                        "Outside start/end the model runs normally with only the base prompt."
                    )},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": (
                        "When regional prompting stops. After this, the model only sees base_prompt and the region "
                        "prompts are gone, so later steps tend to overwrite the regions. Usually leave at 1.0."
                    )},
                ),
                "cross_mask_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": (
                        "How strictly pixels inside a box listen only to that box's prompt (text -> image). "
                        "Multiplied by each region's strength; a product of 1.0 fully blocks the other prompts, "
                        "lower values only discourage them, so prompts bleed more. "
                        "0 turns regional prompts off entirely."
                    )},
                ),
                "self_mask_strength": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": (
                        "How much pixels in one box are kept from looking at pixels in other boxes (image -> image). "
                        "Higher stops features/colours leaking between characters but can cause hard seams "
                        "and a disjointed scene. Also multiplied by region strength. 0 = off."
                    )},
                ),
                "base_ratio": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": (
                        "Blends each step's output with a normal, non-regional pass using only the base prompt. "
                        "0 = pure regional, 1 = regional off. Small values help the image hold together. "
                        "Any value above 0 runs the model twice per active step, so 0.01 costs the same as 0.99."
                    )},
                ),
                "cross_inject_every_n_blocks": (
                    "INT",
                    {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": (
                        "Apply the region text mask on every Nth transformer block (1 = all). "
                        "Skipped blocks still see every region prompt but unmasked, so higher N means more prompt bleed."
                    )},
                ),
                "self_inject_every_n_blocks": (
                    "INT",
                    {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": (
                        "Apply the self-attention separation on every Nth transformer block (1 = all). "
                        "Higher N lets regions mix more, softening seams. Does nothing when self_mask_strength is 0."
                    )},
                ),
                "json_prompt": ("STRING", {"multiline": True, "default": DEFAULT_JSON_PROMPT}),
            },
            "optional": {
                "use_builder": ("BOOLEAN", {"tooltip": "Use built in prompt builder"})
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
        use_builder=False,
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


ANIMA_PATCH_NODE_CLASS_MAPPINGS = {
    "FG_LLLiteCondPreview"   : FG_LLLiteCondPreview,
    "FG_AnimaRegionalPrompt" : FG_AnimaRegionalPrompt,
    "FG_ApplyCosmosReferenceLatent"  : FG_ApplyCosmosReferenceLatent,
}

ANIMA_PATCH_NODE_DISPLAY_NAME_MAPPINGS = {
    "FG_LLLiteCondPreview"          : "⚙️ LLLite Cond Preview",
    "FG_AnimaRegionalPrompt"        : "🗑️ Anima Regional Prompt",
    "FG_ApplyCosmosReferenceLatent" : "🗑️ Anima Apply Cosmos Reference Latent",
}
