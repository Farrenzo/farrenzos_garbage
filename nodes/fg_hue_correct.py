"""
FG_HueCorrect — targeted colour correction in CIELAB.

A LoRA's colour bias is a numerical offset, not a semantic one: every "pink" it
produces lands at roughly the same wrong hue angle. Prompting can't address that.
Rotating the a/b plane can, and because it never touches L, all shading,
highlights and linework survive exactly.

Three modes:

  measure   Change nothing. Report the mean hue angle, chroma and lightness of
            the selection. Run this on a region that SHOULD be pink to find out
            how far off the model actually is, in degrees.

  manual    Apply an explicit hue rotation / chroma scale / lightness shift to
            the selection.

  match     Sample a reference image (your cover art) through its own mask, and
            rotate + scale the target's chroma to land on the reference's mean
            hue and chroma. Internal variation is preserved because chroma is
            scaled, not replaced.

Selection is hue-band x chroma-floor x optional mask, all soft. Always look at
the `selection` output before trusting a correction — a bad selection is the
usual reason these things go wrong.
"""

import math

import torch
import torch.nn.functional as F

_XYZ_FROM_RGB = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                              [0.2126729, 0.7151522, 0.0721750],
                              [0.0193339, 0.1191920, 0.9503041]])
_RGB_FROM_XYZ = torch.tensor([[3.2404542, -1.5371385, -0.4985314],
                              [-0.9692660, 1.8760108, 0.0415560],
                              [0.0556434, -0.2040259, 1.0572252]])
_WHITE = torch.tensor([0.95047, 1.00000, 1.08883])
_EPS, _KAPPA = 216 / 24389, 24389 / 27


def rgb_to_lab(img):
    c = img.clamp(0, 1)
    lin = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).clamp(min=0) ** 2.4)
    xyz = (lin @ _XYZ_FROM_RGB.to(img).T) / _WHITE.to(img)
    f = torch.where(xyz > _EPS, xyz.clamp(min=1e-8) ** (1 / 3), (_KAPPA * xyz + 16) / 116)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return torch.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], dim=-1)


def lab_to_rgb(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    inv = lambda t: torch.where(t ** 3 > _EPS, t ** 3, (116 * t - 16) / _KAPPA)
    xyz = torch.stack([inv(fx), inv(fy), inv(fz)], dim=-1) * _WHITE.to(lab)
    lin = lin = xyz @ _RGB_FROM_XYZ.to(lab).T
    return torch.where(lin <= 0.0031308, lin * 12.92,
                       1.055 * lin.clamp(min=0) ** (1 / 2.4) - 0.055).clamp(0, 1)


def _blur(x, radius):
    if radius <= 0:
        return x
    sigma = max(radius / 2.0, 1e-3)
    k = int(radius) * 2 + 1
    coords = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    pad = k // 2
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), g.view(1, 1, k, 1))


def _selection(lab, hue_center, hue_width, min_chroma, mask, feather):
    """Soft weight in [0,1]: inside the hue band, chromatic enough, inside mask."""
    a, b = lab[..., 1], lab[..., 2]
    chroma = torch.sqrt(a * a + b * b)
    hue = torch.rad2deg(torch.atan2(b, a)) % 360.0

    d = (hue - hue_center).abs()
    d = torch.minimum(d, 360.0 - d)                      # wrap at 0/360
    w = torch.exp(-0.5 * (d / max(hue_width / 2.0, 1e-3)) ** 2)

    # smoothstep off near-neutral pixels: highlights and greys must not rotate
    t = ((chroma - min_chroma) / max(min_chroma, 1e-3)).clamp(0, 1)
    w = w * (t * t * (3 - 2 * t))

    if mask is not None:
        m = mask if mask.ndim == 3 else mask.unsqueeze(0)
        if m.shape[-2:] != w.shape[-2:]:
            m = F.interpolate(m.unsqueeze(1), size=w.shape[-2:],
                              mode="bilinear", align_corners=False).squeeze(1)
        w = w * m.clamp(0, 1)

    if feather > 0:
        w = _blur(w.unsqueeze(1), feather).squeeze(1)
    return w.clamp(0, 1), chroma, hue


def _stats(lab, w):
    """Weight-aware mean hue (circular), chroma and lightness of a selection."""
    a, b = lab[..., 1], lab[..., 2]
    tot = w.sum().clamp(min=1e-6)
    ma, mb = (a * w).sum() / tot, (b * w).sum() / tot
    return (torch.rad2deg(torch.atan2(mb, ma)) % 360.0,
            torch.sqrt(ma * ma + mb * mb),
            (lab[..., 0] * w).sum() / tot,
            float(tot / w.numel()))


class FG_HueCorrect:
    """Rotate a hue band toward a target, in LAB, without touching luminance."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["measure", "manual", "match"], {"default": "measure"}),
                "select_hue": ("FLOAT", {
                    "default": 340.0, "min": 0.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Hue angle to target, LAB degrees. ~25 skin/orange, "
                               "~90 yellow, ~150 green, ~270 blue, ~330 magenta, "
                               "~300-315 is where 'purple pink' tends to sit."}),
                "hue_width": ("FLOAT", {
                    "default": 60.0, "min": 5.0, "max": 360.0, "step": 5.0,
                    "tooltip": "Width of the band. Falloff is gaussian, so this is "
                               "roughly the full width at half strength."}),
                "min_chroma": ("FLOAT", {
                    "default": 8.0, "min": 0.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Ignore pixels less colourful than this. Keeps "
                               "highlights, greys and linework out of the selection."}),
                "feather": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Restrict to a region (hair, a garment)."}),
                "hue_shift": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "manual mode: degrees to rotate the selection."}),
                "chroma_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "manual mode: multiply saturation. Below 1 desaturates."}),
                "lightness_shift": ("FLOAT", {
                    "default": 0.0, "min": -30.0, "max": 30.0, "step": 0.5,
                    "tooltip": "manual mode: L offset, 0-100 scale. Usually leave at 0."}),
                "reference": ("IMAGE", {"tooltip": "match mode: the cover art."}),
                "reference_mask": ("MASK", {
                    "tooltip": "match mode: which part of the cover art to sample. "
                               "Without it the whole reference is sampled through "
                               "the same hue band."}),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original and corrected."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "selection", "info")
    FUNCTION = "correct"
    CATEGORY = "Farrenzo's Garbage/Image/Color"
    DESCRIPTION = "Hue-selective colour correction in LAB. Chroma only; L untouched."

    def correct(self, image, mode, select_hue, hue_width, min_chroma, feather,
                mask=None, hue_shift=0.0, chroma_scale=1.0, lightness_shift=0.0,
                reference=None, reference_mask=None, strength=1.0):
        lab = rgb_to_lab(image[..., :3])
        w, chroma, hue = _selection(lab, select_hue, hue_width, min_chroma, mask, feather)
        src_h, src_c, src_l, cov = _stats(lab, w)

        lines = [f"selection : {cov * 100:.1f}% of pixels (weighted)",
                 f"mean hue  : {src_h:.1f} deg",
                 f"mean chroma: {src_c:.1f}",
                 f"mean L    : {src_l:.1f}"]

        if mode == "measure":
            lines.append("")
            lines.append("measure mode — image unchanged. Compare 'mean hue' against "
                         "where the colour should sit, and use the difference as "
                         "hue_shift in manual mode.")
            info = "\n".join(lines)
            print("[FG_HueCorrect]\n" + info)
            return (image, w.unsqueeze(-1).repeat(1, 1, 1, 3), info)

        if mode == "match":
            if reference is None:
                raise ValueError("match mode needs a reference image.")
            rlab = rgb_to_lab(reference[..., :3])
            rw, _, _ = _selection(rlab, select_hue, hue_width, min_chroma,
                                  reference_mask, feather)
            if float(rw.sum()) < 1e-3:
                raise ValueError(
                    "Nothing selected in the reference. Widen hue_width, lower "
                    "min_chroma, or check reference_mask."
                )
            tgt_h, tgt_c, tgt_l, rcov = _stats(rlab, rw)
            hue_shift = float((tgt_h - src_h + 180.0) % 360.0 - 180.0)
            chroma_scale = float(tgt_c / src_c.clamp(min=1e-3))
            lightness_shift = 0.0
            lines += ["",
                      f"reference : {rcov * 100:.1f}% selected, hue {tgt_h:.1f} deg, "
                      f"chroma {tgt_c:.1f}, L {tgt_l:.1f}",
                      f"derived   : hue_shift {hue_shift:+.1f} deg, "
                      f"chroma_scale {chroma_scale:.2f}"]

        theta = math.radians(hue_shift)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        a, b = lab[..., 1], lab[..., 2]
        na = (a * cos_t - b * sin_t) * chroma_scale
        nb = (a * sin_t + b * cos_t) * chroma_scale
        nL = lab[..., 0] + lightness_shift

        blend = (w * strength).unsqueeze(-1)
        out_lab = torch.stack([
            lab[..., 0] * (1 - blend[..., 0]) + nL * blend[..., 0],
            a * (1 - blend[..., 0]) + na * blend[..., 0],
            b * (1 - blend[..., 0]) + nb * blend[..., 0],
        ], dim=-1)
        out_lab[..., 0] = out_lab[..., 0].clamp(0, 100)

        out = lab_to_rgb(out_lab)
        chk = _stats(rgb_to_lab(out), w)
        lines += ["", f"result    : hue {chk[0]:.1f} deg, chroma {chk[1]:.1f}"]

        info = "\n".join(lines)
        print("[FG_HueCorrect]\n" + info)
        return (out, w.unsqueeze(-1).repeat(1, 1, 1, 3), info)


