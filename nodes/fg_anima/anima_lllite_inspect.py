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

import torch
import torch.nn.functional as F

import folder_paths

from .anima_controlnet_nodes import (
    _build_inpaint_cond_image,
    _prepare_cond_image,
    _prepare_mask,
    _target_cond_hw,
    read_lllite_metadata,
)

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


# ─── runtime firing counters ──────────────────────────────────────────

_STATS = {"fired": 0, "bypassed": 0, "delta": 0.0, "base": 0.0, "n": 0}


def install_lllite_counters(enable=True):
    """Wrap LLLiteModuleDiT.forward to record fired/bypassed and delta magnitude.

    Call once at import time from your pack's __init__. Costs two norms per
    module per step, so leave it off for production runs.
    """
    from .anima_controlnet_nodes import LLLiteModuleDiT

    if getattr(LLLiteModuleDiT, "_fg_counted", False) or not enable:
        return
    original = LLLiteModuleDiT.forward

    def counted(self, x, *a, **kw):
        before = x
        out = original(self, x, *a, **kw)
        if out.shape == before.shape:
            d = (out - before)
            if float(d.abs().max()) < 1e-9:
                _STATS["bypassed"] += 1
            else:
                _STATS["fired"] += 1
                _STATS["delta"] += float(d.norm())
                _STATS["base"] += float(before.norm())
                _STATS["n"] += 1
        else:
            _STATS["bypassed"] += 1
        return out

    LLLiteModuleDiT.forward = counted
    LLLiteModuleDiT._fg_counted = True
    print("[FG] LLLite firing counters installed.")


def report_lllite_stats(reset=True):
    s = _STATS
    total = s["fired"] + s["bypassed"]
    if total == 0:
        print("[FG LLLite] no modules were called at all — the wrapper never ran.")
    else:
        ratio = (s["delta"] / s["base"]) if s["base"] else 0.0
        print(f"[FG LLLite] fired {s['fired']}/{total} "
              f"({100 * s['fired'] / total:.0f}%), bypassed {s['bypassed']}. "
              f"mean |delta|/|x| = {ratio:.4f}")
        if s["fired"] == 0:
            print("            Every module bypassed. Token-count mismatch — "
                  "run FG_LLLiteCondPreview to see the numbers.")
        elif ratio < 0.01:
            print("            Modules fired but the injected delta is under 1% of "
                  "the activations. Raise strength.")
    if reset:
        _STATS.update(fired=0, bypassed=0, delta=0.0, base=0.0, n=0)

