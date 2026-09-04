"""FG_LineworkComposite — put the original panel's linework and text back.

Diffusion cannot reproduce glyph-scale detail: every pixel went through an 8x
VAE and was rebuilt from a latent. So don't ask it to. Colour is low-frequency
(this is why JPEG 4:2:0 works), which means you can generate chroma at half
resolution and recombine it with the original's full-resolution luma.

Feed it the original B&W panel at native size and the coloured output at
whatever size Anima could actually handle. Output is full resolution with
pixel-perfect text.

Modes:
    detail
        frequency separation — low-frequency L from the model (keeps its
        shading and rendering), high-frequency L from the original (lines,
        text, screentone). The default, and usually what you want.
    replace
        L is entirely the original's. Flat cel look, model supplies only
        hue. Use when the model's shading fights the artist's tone.
    off
        L straight from the upscaled model output. For A/B comparison.
"""

import torch
import torch.nn.functional as F

MAX_RESOLUTION = 16384


# ─── sRGB <-> CIELAB (D65), batched, torch-native ─────────────────────

_XYZ_FROM_RGB = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_RGB_FROM_XYZ = torch.tensor([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])
_WHITE = torch.tensor([0.95047, 1.00000, 1.08883])


def _srgb_to_linear(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).clamp(min=0) ** 2.4)


def _linear_to_srgb(c):
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c.clamp(min=0) ** (1 / 2.4) - 0.055)


def rgb_to_lab(img):
    """img: (B, H, W, 3) in [0, 1] -> (B, H, W, 3) as L in [0,100], a/b ~[-128,127]."""
    lin = _srgb_to_linear(img.clamp(0, 1))
    xyz = lin @ _XYZ_FROM_RGB.to(img).T
    xyz = xyz / _WHITE.to(img)

    eps, kappa = 216 / 24389, 24389 / 27
    f = torch.where(xyz > eps, xyz.clamp(min=1e-8) ** (1 / 3), (kappa * xyz + 16) / 116)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return torch.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], dim=-1)


def lab_to_rgb(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200

    eps, kappa = 216 / 24389, 24389 / 27
    def finv(t):
        return torch.where(t ** 3 > eps, t ** 3, (116 * t - 16) / kappa)

    xyz = torch.stack([finv(fx), finv(fy), finv(fz)], dim=-1) * _WHITE.to(lab)
    lin = xyz @ _RGB_FROM_XYZ.to(lab).T
    return _linear_to_srgb(lin).clamp(0, 1)


# ─── separable gaussian blur on a single channel ──────────────────────

def _blur(x, radius):
    """x: (B, 1, H, W). radius in pixels; 0 is a no-op."""
    if radius <= 0:
        return x
    sigma = max(radius / 2.0, 1e-3)
    k = int(radius) * 2 + 1
    coords = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    pad = k // 2
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, g.view(1, 1, 1, k))
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    return F.conv2d(x, g.view(1, 1, k, 1))


# ─── node ─────────────────────────────────────────────────────────────

class FG_LineworkComposite:
    """Recombine full-res original luma with model-generated chroma."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE", {
                    "tooltip": "The source panel at full resolution. Defines output size."}),
                "colored": ("IMAGE", {
                    "tooltip": "Anima's output. Any size; resampled to match."}),
                "mode": (["detail", "replace", "off"], {"default": "detail"}),
                "detail_radius": ("INT", {
                    "default": 6, "min": 0, "max": 64,
                    "tooltip": "Frequency split point, in output pixels. Detail finer "
                               "than this comes from the original. Roughly the upscale "
                               "factor x3 is a good start (2x upscale -> 6)."}),
                "detail_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "How much original high-frequency detail to add back. "
                               "1.0 restores it exactly; above that sharpens."}),
                "chroma_blur": ("INT", {
                    "default": 0, "min": 0, "max": 32,
                    "tooltip": "Blur a/b before recombining. Kills upscale ringing and "
                               "colour fringing along lines. 2-4 helps at 2x."}),
            },
            "optional": {
                "protect_ink": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Force near-black and near-white areas of the original "
                               "to stay neutral. Keeps speech bubbles white and text "
                               "black instead of tinted by colour bleed."}),
                "ink_low": ("FLOAT", {
                    "default": 12.0, "min": 0.0, "max": 50.0, "step": 1.0,
                    "tooltip": "L below this is treated as ink (0-100)."}),
                "ink_high": ("FLOAT", {
                    "default": 94.0, "min": 50.0, "max": 100.0, "step": 1.0,
                    "tooltip": "L above this is treated as paper (0-100)."}),
                "ink_feather": ("FLOAT", {
                    "default": 6.0, "min": 0.0, "max": 30.0, "step": 0.5,
                    "tooltip": "Soft falloff in L units, so edges don't hard-clip."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "composite"
    CATEGORY = "Farrenzo's Garbage/Image/Utils"

    def composite(self, original, colored, mode, detail_radius, detail_strength,
                  chroma_blur, protect_ink=True, ink_low=12.0, ink_high=94.0,
                  ink_feather=6.0):
        B, H, W, _ = original.shape

        col = colored
        if col.shape[0] != B:
            col = col[:1].expand(B, -1, -1, -1) if col.shape[0] == 1 else col[:B]
        if col.shape[1] != H or col.shape[2] != W:
            col = F.interpolate(
                col.permute(0, 3, 1, 2), size=(H, W),
                mode="bicubic", align_corners=False, antialias=False,
            ).clamp(0, 1).permute(0, 2, 3, 1)

        lab_o = rgb_to_lab(original[..., :3])
        lab_c = rgb_to_lab(col[..., :3])

        L_o = lab_o[..., 0:1].permute(0, 3, 1, 2)
        L_c = lab_c[..., 0:1].permute(0, 3, 1, 2)

        if mode == "replace":
            L_out = L_o
        elif mode == "off":
            L_out = L_c
        else:
            # low frequency (shading) from the model, high frequency (lines,
            # text, screentone) from the original
            L_out = _blur(L_c, detail_radius) + detail_strength * (L_o - _blur(L_o, detail_radius))

        ab = lab_c[..., 1:3].permute(0, 3, 1, 2)
        if chroma_blur > 0:
            ab = torch.cat([_blur(ab[:, 0:1], chroma_blur),
                            _blur(ab[:, 1:2], chroma_blur)], dim=1)

        if protect_ink:
            Lo = L_o
            feather = max(ink_feather, 1e-3)
            dark = ((ink_low + feather - Lo) / feather).clamp(0, 1)
            light = ((Lo - (ink_high - feather)) / feather).clamp(0, 1)
            keep = torch.maximum(dark, light)          # 1 where ink or paper
            ab = ab * (1.0 - keep)
            L_out = L_out * (1.0 - keep) + Lo * keep

        L_out = L_out.clamp(0, 100)
        out_lab = torch.cat([L_out, ab], dim=1).permute(0, 2, 3, 1)
        return (lab_to_rgb(out_lab),)
