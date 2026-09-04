"""
Conditioning rebalance for Krea 2 (and any multi-layer-tap text-conditioned model).

Krea 2 conditions on a stack of 12 Qwen3-VL hidden-state layers, packed into a single
(B, seq, 12*D) tensor. The shallow taps encode broad syntax and composition; the deeper
taps carry the fine, specific detail (identity, texture, precise attributes) that can be
under-represented after alignment training. This node reshapes that tensor to expose the
layer axis, applies a per-layer gain (so you can emphasise the taps you care about),
optionally holds the overall magnitude constant, then applies a global multiplier — all
while leaving masks / pooled outputs untouched.

Enhanced fork of nova452/ComfyUI-ConditioningKrea2Rebalance (Apache-2.0). The original
introduced per-layer weighting but then applies a single global multiplier to every tap at
once, amplifying the whole conditioning tensor — which is what degrades likeness, prompt
adherence and colour when pushed past ~1x. This fork defaults to RMS-renormalised per-layer
rebalancing: shift the tap ratios, hold the magnitude, recover the detail without trashing
the model. It also adds named presets, hardened parsing and full type hints. The original
node's behaviour is still available with renormalize=False, multiplier=4.0.

Copyright 2026 Hu White. Licensed under the Apache License, Version 2.0.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch

# ---------------------------------------------------------------------------
# Preset per-layer weight profiles (12 taps, matching Krea 2's Qwen3-VL stack).
# Tap order matches the model's layer aggregation (shallow -> deep).
# ---------------------------------------------------------------------------
PRESET_WEIGHTS: dict = {
    # The classic nova452 profile: gentle on shallow taps, strong boost on deep taps.
    "balanced": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0],
    # Maximum fine-detail adherence — pushes the deepest taps harder.
    "detail": [0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.2, 3.0, 6.0, 1.5, 5.0, 1.2],
    # A light touch for when the base model only needs a small nudge.
    "subtle": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 1.0, 1.5, 1.0],
    # No per-layer change — pair with multiplier > 1 for a clean global conditioning boost.
    "uniform": [1.0] * 12,
}

PRESET_NAMES = ["balanced", "detail", "subtle", "uniform", "custom"]


def parse_weights(text: str) -> List[float]:
    """Parse a comma/semicolon-separated weight string into a list of floats.

    Raises ValueError on unparseable input or fewer than 2 values. Tap-count alignment
    against the actual conditioning tensor is handled gracefully downstream.
    """
    if text is None:
        raise ValueError("per_layer_weights is empty.")
    parts = [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip() != ""]
    try:
        vals = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"per_layer_weights has a non-numeric entry: {exc}") from exc
    if len(vals) < 2:
        raise ValueError("per_layer_weights needs at least 2 values.")
    return vals


def _rms(t: torch.Tensor) -> torch.Tensor:
    """Root-mean-square over every dim except the batch (dim 0). Returns shape (B,)."""
    return t.pow(2).mean(dim=tuple(range(1, t.dim()))).sqrt()


def _scale_cond_tensor(
    t: torch.Tensor,
    multiplier: float,
    per_layer_weights: Optional[Sequence[float]] = None,
    renormalize: bool = False,
) -> torch.Tensor:
    """Scale a conditioning tensor with optional per-layer weighting and RMS renormalisation.

    The tensor arrives as (B, seq, N*D) — N taps flattened into the feature dim. When
    per_layer_weights is given we reshape to (B, seq, N, D), apply a per-tap gain, flatten
    back, and (optionally) renormalise so the output RMS matches the input RMS — letting you
    rebalance the *ratios* between taps without inflating the overall conditioning magnitude.
    A final global multiplier is always applied. If the feature dim isn't divisible by N (not a
    stacked-tap tensor), it falls back to a uniform scale so the node never breaks a graph.
    """
    if per_layer_weights is None or len(per_layer_weights) <= 1:
        return t * multiplier

    flat = t.shape[-1]
    n_layers = len(per_layer_weights)
    if flat % n_layers != 0:
        return t * multiplier

    orig_dtype = t.dtype
    ref_rms = _rms(t.float()) if renormalize else None
    t = t.float()
    t = t.view(*t.shape[:-1], n_layers, flat // n_layers)
    gains = torch.tensor(list(per_layer_weights), dtype=t.dtype, device=t.device)
    t = t * gains.view(*([1] * (t.dim() - 2)), n_layers, 1)
    t = t.view(*t.shape[:-2], flat)

    if renormalize and ref_rms is not None:
        new_rms = _rms(t).clamp_min(1e-8)
        t = t * (ref_rms / new_rms).view(-1, *([1] * (t.dim() - 1)))

    return t.to(orig_dtype) * multiplier


def scale_conditioning(
    structure,
    multiplier: float,
    per_layer_weights: Optional[Sequence[float]] = None,
    renormalize: bool = False,
):
    """Recursively scale every conditioning tensor in a ComfyUI CONDITIONING structure.

    Leaves masks, pooled outputs, and any non-tensor payloads intact.
    """
    if isinstance(structure, list):
        out = []
        for item in structure:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[0], torch.Tensor)
                and isinstance(item[1], dict)
            ):
                cond_t, extras = item
                out.append(
                    [
                        _scale_cond_tensor(cond_t, multiplier, per_layer_weights, renormalize),
                        dict(extras),
                    ]
                )
            else:
                out.append(
                    scale_conditioning(item, multiplier, per_layer_weights, renormalize)
                )
        return out
    if isinstance(structure, torch.Tensor):
        return _scale_cond_tensor(structure, multiplier, per_layer_weights, renormalize)
    if isinstance(structure, dict):
        return {
            k: scale_conditioning(v, multiplier, per_layer_weights, renormalize)
            for k, v in structure.items()
        }
    return structure


class ConditioningKrea2Rebalance:
    """Per-layer conditioning reweighting for Krea 2 (and multi-layer-tap models)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "preset": (
                    PRESET_NAMES,
                    {
                        "default": "balanced",
                        "tooltip": "Named per-layer weight profile. Use 'custom' to supply your own via per_layer_weights.",
                    },
                ),
                "per_layer_weights": (
                    "STRING",
                    {
                        "default": ", ".join(str(w) for w in PRESET_WEIGHTS["balanced"]),
                        "multiline": False,
                        "tooltip": "Comma-separated gains, one per conditioning tap (12 for Krea 2). Used only when preset = 'custom'.",
                    },
                ),
                "multiplier": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -1000.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "tooltip": "Global gain applied after per-layer weighting. Keep ~1.0 with renormalize on for quality-preserving rebalance; values >1 amplify the whole tensor and can oversaturate.",
                    },
                ),
                "renormalize": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Hold the input RMS after per-layer weighting so tap ratios change but overall magnitude does not. This is the quality-preserving mode and is on by default.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "rebalance"
    CATEGORY = "Farrenzo's Garbage/Conditioning"
    DESCRIPTION = "Rebalance Krea 2's multi-layer text conditioning per tap, then apply a global gain."

    def rebalance(
        self,
        conditioning,
        preset: str = "balanced",
        per_layer_weights: str = "",
        multiplier: float = 1.0,
        renormalize: bool = True,
    ):
        if preset == "custom":
            weights = parse_weights(per_layer_weights)
        else:
            weights = PRESET_WEIGHTS.get(preset, PRESET_WEIGHTS["balanced"])
        return (scale_conditioning(conditioning, multiplier, weights, renormalize),)