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
import math
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.ldm.common_dit
import comfy.model_management
import comfy.patcher_extension
from comfy.ldm.flux.layers import timestep_embedding

from ._fg_helperfunctions import log


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


class FG_ConditioningKrea2Rebalance:
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


# ===========================================================================
# Krea2 Edit — in-context edit forward
# ===========================================================================
"""
ComfyUI's native Krea2 `_forward` is text-to-image only: it builds `[text | target]`.
The krea2_edit LoRA (trained in ai-toolkit) also needs the APPEARANCE path: the
VAE-encoded source image(s) prepended as clean tokens, told apart from the noisy target
purely by the RoPE frame index (refs = 1..N, target = 0). This node wraps the diffusion
model's forward and rebuilds the sequence as `[text | ref1 | ref2 | target]`, keeping only
the target tokens out — mirroring ai-toolkit's `predict_velocity_edit`.

The SEMANTIC path (same images fed to Qwen3-VL with the prompt) lives in
FG_CLIPTextEncode. Feed it the same images in the same order.

Wiring:
    UNETLoader -> LoraLoaderModelOnly -> FG_Krea2EditModelPatch(model, vae, image1, image2,
                                                                width, height)
    FG_Krea2EditModelPatch.Model        -> KSampler.model
    FG_Krea2EditModelPatch.Empty Latent -> KSampler.latent_image

Why this is fast: width/height are known at node-execution time, so every reference is
fitted and VAE-encoded ONCE here, before sampling. The old node only learned the target
size inside the sampler, so the VAE got loaded mid-sampling, evicted part of the diffusion
model, and every remaining step streamed weights from CPU.
"""

# The trainer snaps genuine-AR-mismatch refs to this pixel grid (VAE x8 * patch x2).
# Train/infer geometry must be identical or a seam appears at the ref margin.
TRAINER_GRID_PX = 16
# Near-matched AR: fill the target exactly with a minimal center-crop instead of leaving
# 1-2 token margins (those margins caused edge duplication).
NEAR_MATCH_CROP_TOLERANCE = 0.08

REF_POSITIONS = ["aligned (trained)", "beside (right of target)", "bag of features"]


def _imgids(bs, frame, h_, w_, device):
    ids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)


def _imgids_centered(bs, frame, gh, gw, th, tw, device):
    """Stride-1 positions for a ref grid (gh, gw) centered inside the target grid (th, tw).
    Refs are resampled in pixel space to the target's grid density, so stride-1 is correct
    by construction. Fractional centers are fine: RoPE is continuous."""
    off_h, off_w = max(0.0, (th - gh) / 2), max(0.0, (tw - gw) / 2)
    ids = torch.zeros(gh, gw, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = (torch.arange(gh, device=device, dtype=torch.float32) + off_h)[:, None]
    ids[..., 2] = (torch.arange(gw, device=device, dtype=torch.float32) + off_w)[None, :]
    return ids.reshape(1, gh * gw, 3).repeat(bs, 1, 1)


def _place_refs(ref_ids, modes, h_, w_):
    """Per-ref placement on the h/w RoPE axes (frame axis untouched).
    aligned (trained): ref token (h, w) sits on target token (h, w), how the models were trained.
    beside: ref shifted to the right of the canvas (and earlier refs); identity flows, layout pin is gone.
    bag of features (experimental): every ref token at the target's center; content only, no layout."""
    out, x0 = [], float(w_)
    for ids, mode in zip(ref_ids, modes):
        if mode.startswith("aligned"):
            out.append(ids)
            continue
        ids = ids.clone()
        if mode.startswith("beside"):
            ids[..., 2] = ids[..., 2] - ids[..., 2].min() + x0
            x0 = float(ids[..., 2].max()) + 1.0
        elif mode.startswith("bag"):
            ids[..., 1] = (h_ - 1) / 2.0
            ids[..., 2] = (w_ - 1) / 2.0
        else:
            raise ValueError(f"unknown reference position {mode!r}")
        out.append(ids)
    return out


def _to_4d(v):
    """(B,C,T,H,W) -> (B*T,C,H,W); 4D passes through. Images use T=1."""
    if v.ndim == 5:
        b, c, t, h, w = v.shape
        return v.reshape(b * t, c, h, w)
    return v


def _fit_encode_image(image, vae, lat_h, lat_w, compression):
    """Fit a reference IMAGE to the target grid in PIXEL space, then VAE-encode.
    Never resizes latents (that softens them). Geometry matches the trainer's _fit_prep."""
    px_h, px_w = lat_h * compression, lat_w * compression
    img = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
    ih, iw = img.shape[-2:]
    sc = min(px_h / ih, px_w / iw)
    tol = NEAR_MATCH_CROP_TOLERANCE
    if ih * sc >= px_h * (1 - tol) and iw * sc >= px_w * (1 - tol):
        # Near-matched AR: minimal center-crop, fill the target exactly.
        s = max(px_h / ih, px_w / iw)
        ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
        y0, x0 = (ih - ch) // 2, (iw - cw) // 2
        img = img[..., y0:y0 + ch, x0:x0 + cw]
        nh, nw = px_h, px_w
    else:
        # Genuine AR mismatch: fit inside, snapped down to the trainer grid, capped at
        # the target's grid. Center-crop the source so the fitted axis lands on the grid
        # at scale `sc` exactly (resizing straight to the floor squashes content and
        # produces a doubled band at the ref edge).
        g = TRAINER_GRID_PX
        nh = min(max(g, int(ih * sc) // g * g), max(g, px_h // g * g))
        nw = min(max(g, int(iw * sc) // g * g), max(g, px_w // g * g))
        ch, cw = min(ih, max(1, int(round(nh / sc)))), min(iw, max(1, int(round(nw / sc))))
        y0, x0 = (ih - ch) // 2, (iw - cw) // 2
        img = img[..., y0:y0 + ch, x0:x0 + cw]
    img = F.interpolate(img.float(), size=(nh, nw), mode="bicubic", antialias=True)
    return vae.encode(img.movedim(1, -1)[..., :3].clamp(0, 1))


def _ref_attn_bias(boosts, boost_mask, txtlen, slens, tgtlen, grids, device, dtype):
    """Additive attention-logit bias on [text | refs... | target]: log(boost) on
    target->ref attention. boost_mask (ref-image pixel space) restricts the LAST ref's
    boost to a region, e.g. the face."""
    nsrc = len(slens)
    offs = [txtlen]
    for sl in slens:
        offs.append(offs[-1] + sl)
    rows0 = offs[-1]
    L = rows0 + tgtlen
    bias = torch.zeros(1, 1, L, L, device=device, dtype=dtype)
    for i, b in enumerate(boosts):
        if b == 1.0:
            continue
        off, sl = offs[i], slens[i]
        if boost_mask is not None and i == nsrc - 1:
            mask = boost_mask[:1]
            if mask.ndim == 2:
                mask = mask[None]
            mask = F.interpolate(mask[None].float(), size=grids[i], mode="area")[0, 0]
            cols = off + torch.nonzero(mask.reshape(-1) > 0.5, as_tuple=True)[0].to(device)
        else:
            cols = torch.arange(off, off + sl, device=device)
        bias[:, :, rows0:, cols] = math.log(max(b, 1e-4))
    return bias


def _log_once(state, key, message, message_type=None):
    if key in state["logged"]:
        return
    state["logged"].add(key)
    if message_type:
        log(message, message_type=message_type)
    else:
        log(message)


def krea2_edit_forward(m, x, timesteps, context, refs, transformer_options,
                       attention_boosts, positions, boost_mask, state):
    """Krea2 SingleStreamDiT._forward with reference block(s) prepended.

    m        : the SingleStreamDiT (LoRA-patched at sample time)
    x        : (B,C,H,W) or (B,C,T,H,W) noisy TARGET latent
    refs     : list of clean reference latents, already fitted, processed and on x's
               device/dtype, batch 1 — in training order (frames 1..N)
    context  : (B, seq, 12*D) — the 12-layer Qwen3-VL stack
    """
    patch = m.patch

    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
    x = _to_4d(x)
    bs, c, H_orig, W_orig = x.shape

    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch), padding_mode="replicate")
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch

    srcs = []
    for ref in refs:
        ref = comfy.ldm.common_dit.pad_to_patch_size(ref[:1], (patch, patch), padding_mode="replicate")
        srcs.append(ref.expand(bs, *ref.shape[1:]))
    grids = [(s_.shape[-2] // patch, s_.shape[-1] // patch) for s_ in srcs]

    context = m._unpack_context(context)

    tgt_img = m.first(rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
    src_imgs = [m.first(rearrange(s_, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
                for s_ in srcs]

    t = m.tmlp(timestep_embedding(timesteps, m.tdim).unsqueeze(1).to(tgt_img.dtype))
    tvec = m.tproj(t)

    context = m.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = m.txtmlp(context)

    txtlen, tgtlen = context.shape[1], tgt_img.shape[1]
    slens = [si.shape[1] for si in src_imgs]
    srclen = sum(slens)
    combined = torch.cat([context] + src_imgs + [tgt_img], dim=1)  # [text | refs... | target]
    device = combined.device

    _log_once(state, ("grids", tuple(grids), h_, w_),
              f"[krea2edit] reference grids {grids} centered in target grid ({h_}, {w_})")

    # Positions depend only on shapes, so build the RoPE frequencies once per shape
    # (cond and uncond can differ in text length, hence the key).
    freqs_key = (bs, txtlen, h_, w_, tuple(grids), str(device))
    freqs = state["freqs"].get(freqs_key)
    if freqs is None:
        ref_ids = [_imgids_centered(bs, i + 1, gh, gw, h_, w_, device) for i, (gh, gw) in enumerate(grids)]
        ref_ids = _place_refs(ref_ids, positions, h_, w_)
        pos = torch.cat(
            [torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)]  # text @ 0
            + ref_ids
            + [_imgids(bs, 0, h_, w_, device)],                                # target frame 0
            dim=1)
        freqs = m.pe_embedder(pos)
        state["freqs"][freqs_key] = freqs

    attn_bias = None
    if any(b != 1.0 for b in attention_boosts):
        bias_key = (txtlen, tuple(slens), tgtlen, str(device), combined.dtype)
        attn_bias = state["attn_bias"].get(bias_key)
        if attn_bias is None:
            attn_bias = _ref_attn_bias(attention_boosts, boost_mask, txtlen, slens, tgtlen,
                                       grids, device, combined.dtype)
            state["attn_bias"][bias_key] = attn_bias

    for block in m.blocks:
        combined = block(combined, tvec, freqs, attn_bias, transformer_options=transformer_options)

    final = m.last(combined, t)
    out = final[:, txtlen + srclen: txtlen + srclen + tgtlen, :]  # target tokens only
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_, w=w_, ph=patch, pw=patch, c=m.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, m.channels, H_orig, W_orig).movedim(1, 2)
    return out


class FG_Krea2EditModelPatch:
    """Krea2 edit: adds reference image(s) as in-context tokens and outputs a matching empty latent."""

    @classmethod
    def INPUT_TYPES(cls):
        boost_tip = ("Reference-fidelity dial: multiplies how strongly the output attends to this image. "
                     "1.0 = off, >1 pulls harder toward its appearance, <1 loosens. Optimal value is model-specific. "
                     "Values other than 1.0 force masked attention, which is slower.")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Krea2 model with the krea2_edit LoRA applied."}),
                "vae": ("VAE", {"tooltip": "Encodes the reference images into latents. Runs once, before sampling."}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": TRAINER_GRID_PX,
                                  "tooltip": "Output width in pixels (snapped to 16)."}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": TRAINER_GRID_PX,
                                   "tooltip": "Output height in pixels (snapped to 16)."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "image1": ("IMAGE", {"tooltip": "First reference. In two-image workflows this is the SCENE (training order: scene, subject). "
                                                "Feed the same images, same order, to the text encoder."}),
                "image2": ("IMAGE", {"tooltip": "Second reference: the SUBJECT in two-image workflows."}),
                "image1_attention_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05, "round": 0.001, "tooltip": boost_tip}),
                "image2_attention_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05, "round": 0.001, "tooltip": boost_tip}),
                "image1_position": (REF_POSITIONS, {"default": REF_POSITIONS[0],
                    "tooltip": "Where image1 sits in position space. aligned (trained) = pinned onto the output layout (default). "
                               "beside = next to the canvas; identity flows but the layout is free to change. "
                               "bag of features = experimental, content without layout."}),
                "image2_position": (REF_POSITIONS, {"default": REF_POSITIONS[0],
                    "tooltip": "Same choice for image2. To put a person into a scene: image1 (scene) aligned + image2 (person) beside."}),
                "attention_boost_mask": ("MASK", {"tooltip": "Optional region (in that image's pixel space) to limit the attention boost "
                                                             "of the LAST connected image, e.g. the face. Empty = whole image."}),
            },
        }

    RETURN_TYPES = ("MODEL", "LATENT")
    RETURN_NAMES = ("Model", "Empty Latent")
    OUTPUT_TOOLTIPS = ("Patched model. Connect to KSampler.model.",
                       "Empty latent at width x height. Connect to KSampler.latent_image so the "
                       "pre-encoded references match and no VAE work happens during sampling.")
    FUNCTION = "patch"
    CATEGORY = "Farrenzo's Garbage/Krea2"
    DESCRIPTION = ("Krea2 edit: fits and VAE-encodes up to two reference images once, before sampling, "
                   "injects them as in-context tokens, and outputs a matching empty latent.")

    def patch(self, model, vae, width, height, batch_size=1, image1=None, image2=None,
              image1_attention_boost=1.0, image2_attention_boost=1.0,
              image1_position=REF_POSITIONS[0], image2_position=REF_POSITIONS[0],
              attention_boost_mask=None, **_future):
        if _future:
            log(f"[krea2edit] workflow provides inputs this node version does not know "
                f"({', '.join(sorted(_future))}). Continuing without them.", message_type="warning")

        compression = vae.spacial_compression_encode() if hasattr(vae, "spacial_compression_encode") else 8
        width = max(TRAINER_GRID_PX, width // TRAINER_GRID_PX * TRAINER_GRID_PX)
        height = max(TRAINER_GRID_PX, height // TRAINER_GRID_PX * TRAINER_GRID_PX)
        lat_h, lat_w = height // compression, width // compression

        latent_channels = model.model.latent_format.latent_channels
        empty_latent = {"samples": torch.zeros([batch_size, latent_channels, lat_h, lat_w],
                                               device=comfy.model_management.intermediate_device())}

        refs = [(img, boost, position) for img, boost, position in (
            (image1, image1_attention_boost, image1_position),
            (image2, image2_attention_boost, image2_position),
        ) if img is not None]
        if not refs:
            log("[krea2edit] no reference images connected; returning the model unpatched.", message_type="warning")
            return (model, empty_latent)

        images = [r[0] for r in refs]
        attention_boosts = [float(r[1]) for r in refs]
        positions = [r[2] for r in refs]
        diffusion_model = model.model
        storage_device = comfy.model_management.intermediate_device()

        def encode_refs(lh, lw):
            # Krea2's VAE returns 5D (B,C,T,H,W); flatten to 4D once here, not every step.
            return [_to_4d(diffusion_model.process_latent_in(_fit_encode_image(img, vae, lh, lw, compression))).to(storage_device)
                    for img in images]

        state = {
            "encoded": {},    # (lat_h, lat_w) -> processed ref latents on the storage device
            "on_device": {},  # (lat_h, lat_w, device, dtype) -> same, on the sampling device
            "freqs": {},
            "attn_bias": {},
            "logged": set(),
        }

        # The whole point: encode NOW, at node-execution time. The sampler loads the
        # diffusion model afterwards and evicts the VAE, not the other way round.
        log(f"[krea2edit] encoding {len(images)} reference(s) at {width}x{height}px before sampling")
        state["encoded"][(lat_h, lat_w)] = encode_refs(lat_h, lat_w)

        def wrapper(executor, x, timesteps, context, *wargs, **kwargs):
            # ComfyUI signature drift: transformer_options is either a kwarg or the
            # trailing dict positional (newer builds insert ref_latents before it).
            # Native ref_latents are ignored; this patch supplies its own refs.
            transformer_options = kwargs.pop("transformer_options", None)
            if transformer_options is None:
                transformer_options = {}
                for a in reversed(wargs):
                    if isinstance(a, dict):
                        transformer_options = a
                        break

            xx = _to_4d(x)
            size_key = (xx.shape[-2], xx.shape[-1])
            if size_key not in state["encoded"]:
                _log_once(state, ("mid_sample_encode", size_key),
                          f"[krea2edit] sampling at {size_key[1] * compression}x{size_key[0] * compression}px but the node "
                          f"was set to {width}x{height}px. Encoding references mid-sampling (slow). "
                          f"Feed KSampler this node's Empty Latent, or match width/height.",
                          message_type="warning")
                state["encoded"][size_key] = encode_refs(*size_key)

            device_key = size_key + (str(x.device), x.dtype)
            refs_on_device = state["on_device"].get(device_key)
            if refs_on_device is None:
                refs_on_device = [r.to(x.device, x.dtype) for r in state["encoded"][size_key]]
                state["on_device"] = {device_key: refs_on_device}  # keep a single device copy

            return krea2_edit_forward(executor.class_obj, x, timesteps, context, refs_on_device,
                                      transformer_options, attention_boosts, positions,
                                      attention_boost_mask, state)

        patched = model.clone()
        to = patched.model_options.setdefault("transformer_options", {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "krea2_edit", wrapper, to
        )
        return (patched, empty_latent)


KREA2_NODE_CLASS_MAPPINGS = {
    "FG_ConditioningKrea2Rebalance": FG_ConditioningKrea2Rebalance,
    "FG_Krea2EditModelPatch": FG_Krea2EditModelPatch,
}
KREA2_NODE_DISPLAY_NAME_MAPPINGS = {
    "FG_Krea2EditModelPatch": "🗑️ Krea2 Edit Model Patch",
    "FG_ConditioningKrea2Rebalance": "🎛️ Krea 2 Conditioning Control",
}