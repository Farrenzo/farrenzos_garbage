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

import cv2
import numpy as np
import comfy.utils



_XYZ_FROM_RGB = torch.tensor(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ]
)
_RGB_FROM_XYZ = torch.tensor(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ]
)
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


"""
FG_ImageForensics — show what is in a "blank" area that your eyes can't see.

A value of 252 against a background of 255 is invisible on any display and is
still perfectly good signal to a VAE. Erased artwork, JPEG ringing and screentone
moire all live in that band. If a LoRA keeps drawing sweat drops that "aren't
there", the first thing to establish is whether they really aren't.

Outputs
  amplified   The chosen value band stretched to full range. Set the band to
              0.90-1.00 to see what is hiding on white paper.
  highpass    Image minus its own blur, centred on grey. The same thing your
              lineart extractor does, but with numbers attached.
  alpha       The alpha channel if the image has one, else black.

The report covers quantisation, dynamic range, how much signal sits in the band,
and a screentone estimate — high-frequency energy that will alias into visible
blobs when you downscale.
"""

def _blur(x, radius):
    if radius <= 0:
        return x
    sigma = max(radius / 2.0, 1e-3)
    k = int(radius) * 2 + 1
    c = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    p = k // 2
    x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, p, p), mode="reflect"), g.view(1, 1, k, 1))


class FG_ImageForensics:
    """Reveal sub-visible content and measure screentone risk before downscaling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "band_low": ("FLOAT", {
                    "default": 0.90, "min": 0.0, "max": 1.0, "step": 0.005,
                    "tooltip": "Bottom of the value band to stretch. 0.90 for white "
                               "paper, 0.00 with band_high 0.10 for black areas."}),
                "band_high": ("FLOAT", {
                    "default": 1.00, "min": 0.0, "max": 1.0, "step": 0.005}),
                "highpass_radius": ("INT", {
                    "default": 4, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Blur radius subtracted from the image. Small values "
                               "catch screentone; large ones catch soft erasures."}),
                "highpass_gain": ("FLOAT", {
                    "default": 8.0, "min": 1.0, "max": 64.0, "step": 1.0}),
            },
            "optional": {
                "target_megapixels": ("FLOAT", {
                    "default": 1.5, "min": 0.1, "max": 20.0, "step": 0.1,
                    "tooltip": "The size you plan to downscale to. Used to estimate "
                               "how much high-frequency detail will alias."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("amplified", "highpass", "alpha", "report")
    FUNCTION = "inspect"
    CATEGORY = "Farrenzo's Garbage/Image/Utils"
    DESCRIPTION = "Amplify sub-visible signal and estimate screentone aliasing risk."

    def inspect(self, image, band_low, band_high, highpass_radius, highpass_gain,
                target_megapixels=1.5):
        img = image[0]
        H, W, C = img.shape
        rgb = img[..., :3]

        alpha_out = torch.zeros(1, H, W, 3)
        has_alpha = C >= 4
        if has_alpha:
            alpha_out = img[..., 3:4].repeat(1, 1, 3).unsqueeze(0)

        lum = (rgb * torch.tensor([0.2126, 0.7152, 0.0722])).sum(-1)

        # quantisation: how many distinct levels, and is it 8-bit on a 0-255 grid?
        q = torch.unique((lum * 255).round())
        levels = int(torch.unique(lum).numel())
        on_grid = bool(torch.allclose(lum, (lum * 255).round() / 255, atol=1e-4))

        lo, hi = min(band_low, band_high), max(band_low, band_high)
        in_band = ((lum >= lo) & (lum <= hi))
        band_frac = float(in_band.float().mean()) * 100
        band_vals = lum[in_band]
        band_levels = int(torch.unique(band_vals).numel()) if band_vals.numel() else 0

        # Level counting alone misses a hard-edged erasure. What matters is whether
        # the off-background pixels form a shape, so measure how many pixels differ
        # from the band's most common value and how clustered they are.
        off_frac, off_px = 0.0, 0
        if band_vals.numel():
            v, cnt = torch.unique(band_vals, return_counts=True)
            modal = v[cnt.argmax()]
            off = in_band & (lum - modal).abs().gt(1.0 / 512)
            off_px = int(off.sum())
            off_frac = off_px / max(1, int(in_band.sum()))

        amplified = ((rgb - lo) / max(hi - lo, 1e-6)).clamp(0, 1).unsqueeze(0)

        l4 = lum.view(1, 1, H, W)
        hp = l4 - _blur(l4, highpass_radius)
        hp_energy = float(hp.abs().mean())
        highpass = (hp * highpass_gain + 0.5).clamp(0, 1).squeeze(0).squeeze(0)
        highpass = highpass.unsqueeze(-1).repeat(1, 1, 3).unsqueeze(0)

        # Screentone / aliasing estimate: energy above the Nyquist limit of the
        # planned downscale is what folds back as moire.
        mp = (H * W) / 1e6
        scale = (target_megapixels / mp) ** 0.5 if mp > 0 else 1.0
        nyq_radius = max(1, int(round(1.0 / max(scale, 1e-3))))
        above_nyq = float((l4 - _blur(l4, nyq_radius)).abs().mean())
        risk = above_nyq / max(hp_energy, 1e-6)

        lines = [
            f"size        : {W}x{H}  ({mp:.2f} MP), {C} channels"
            + ("  ALPHA PRESENT" if has_alpha else ""),
            f"levels      : {levels} distinct luminance values"
            + ("  (on the 8-bit grid)" if on_grid else "  (NOT 8-bit — 16-bit or resampled)"),
            f"range       : {float(lum.min()):.4f} .. {float(lum.max()):.4f}",
            "",
            f"band {lo:.2f}-{hi:.2f}: {band_frac:.1f}% of pixels, "
            f"{band_levels} distinct levels inside it",
        ]

        if band_frac > 5 and off_px > 64:
            lines.append(
                f"  -> {off_px:,} pixels ({off_frac * 100:.3f}% of the band) differ "
                f"from the background value across {band_levels} levels. There IS "
                f"content here you cannot see. Look at 'amplified'. If the shapes "
                f"land where the model keeps drawing things, it is reading real "
                f"residue, not hallucinating."
            )
        elif band_frac > 5:
            lines.append(
                "  -> the band is flat to within 1/512. Whatever the model is adding "
                "is not coming from residue at this luminance."
            )

        lines += [
            "",
            f"highpass r={highpass_radius}: mean |detail| = {hp_energy:.5f}",
            f"downscale to {target_megapixels:.1f} MP = {scale:.3f}x "
            f"(Nyquist radius ~{nyq_radius}px)",
            f"energy above Nyquist  : {above_nyq:.5f}  ({risk * 100:.0f}% of total detail)",
        ]

        if risk > 0.5:
            lines.append(
                "  -> HIGH aliasing risk. Over half the detail is finer than the "
                "downscale can represent. With bicubic or bilinear this folds into "
                "low-frequency blobs, and a model reads blobs as shading. Use area "
                "resampling, or blur to the Nyquist radius before downscaling."
            )
        elif risk > 0.25:
            lines.append("  -> moderate aliasing risk. Prefer area resampling.")
        else:
            lines.append("  -> low aliasing risk at this target size.")

        report = "\n".join(lines)
        print("[FG_ImageForensics]\n" + report)
        return (amplified, highpass, alpha_out, report)


def color_blend(bw_layer,color_layer):
    # Convert the color layer to LAB color space
    color_lab = cv2.cvtColor(color_layer, cv2.COLOR_BGR2Lab)
    # Convert the black and white layer to grayscale
    bw_layer_gray = cv2.cvtColor(bw_layer, cv2.COLOR_BGR2GRAY)
    # Replace the luminosity (L) channel in the color image with the black and white luminosity
    _, color_a, color_b = cv2.split(color_lab)
    blended_lab = cv2.merge((bw_layer_gray, color_a, color_b))
    # Convert the blended LAB image back to BGR color space
    blended_result = cv2.cvtColor(blended_lab, cv2.COLOR_Lab2BGR)
    return blended_result

class FG_ColorBlend:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bw_layer": ("IMAGE",),
                "color_layer": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_blending_mode"

    CATEGORY = "stability/image/postprocessing"

    def color_blending_mode(self, bw_layer, color_layer):
        if bw_layer.shape[0] < color_layer.shape[0]:
            bw_layer = bw_layer.repeat(color_layer.shape[0], 1, 1, 1)[:color_layer.shape[0]]
        if bw_layer.shape[0] > color_layer.shape[0]:
            color_layer = color_layer.repeat(bw_layer.shape[0], 1, 1, 1)[:bw_layer.shape[0]]

        batch_size, height, width, _ = bw_layer.shape
        tensor_output = torch.empty_like(bw_layer)

        image1 = bw_layer.cpu()
        image2 = color_layer.cpu()
        if image1.shape != image2.shape:
            image2 = image2.permute(0, 3, 1, 2)
            image2 = comfy.utils.common_upscale(image2, image1.shape[2], image1.shape[1], upscale_method='bicubic', crop='center')
            image2 = image2.permute(0, 2, 3, 1)
        image1  = (image1 * 255).to(torch.uint8).numpy()
        image2 = (image2 * 255).to(torch.uint8).numpy()

        for i in range(batch_size):
            blend = color_blend(image1[i],image2[i])
            blend = np.stack([blend])
            tensor_output[i:i+1] = (torch.from_numpy(blend.transpose(0, 3, 1, 2))/255.0).permute(0, 2, 3, 1)

        return (tensor_output,)


"""
LAB Color Transfer - ComfyUI Custom Node
Recolors TARGET to match the color statistics of REFERENCE.
"""

def tensor_to_numpy_bgr(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.squeeze(0).cpu().numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def numpy_bgr_to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
    return tensor.unsqueeze(0)


def lab_color_transfer(
    target_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    strength: float = 1.0,
    match_l: bool = True,
    match_a: bool = True,
    match_b: bool = True
) -> np.ndarray:
    """Recolor target_bgr to match the LAB statistics of reference_bgr."""
    tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    ref_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

    result_lab = tgt_lab.copy()
    channels = {0: match_l, 1: match_a, 2: match_b}

    for ch, do_transfer in channels.items():
        if not do_transfer:
            continue
        tgt_mean = tgt_lab[:, :, ch].mean()
        tgt_std  = tgt_lab[:, :, ch].std()
        ref_mean = ref_lab[:, :, ch].mean()
        ref_std  = ref_lab[:, :, ch].std()

        if tgt_std < 1e-6:
            continue

        transferred = (tgt_lab[:, :, ch] - tgt_mean) / tgt_std * ref_std + ref_mean
        result_lab[:, :, ch] = tgt_lab[:, :, ch] * (1.0 - strength) + transferred * strength

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


class FG_LABColorTransfer:
    def __init__(self):
        self.NODE_NAME = "LAB Color Transfer"

    CATEGORY = "Farrenzo's Garbage/Image/Utils"
    FUNCTION = "transfer"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    DESCRIPTION = "Recolors TARGET to match the color statistics of REFERENCE."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("IMAGE", {"tooltip": "Image to recolor"}),
                "reference": ("IMAGE", {"tooltip": "Image to copy colors from"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "0 = no change, 1 = full transfer, >1 = over-correct"}),
            },
            "optional": {
                "match_luminance": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Transfer the L (brightness/luminance) channel"
                }),
                "match_green_red": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Transfer the A (green-red) channel"
                }),
                "match_blue_yellow": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Transfer the B (blue-yellow) channel"
                }),
                "mask": ("MASK", {
                    "tooltip": "Optional mask (1 = apply transfer, 0 = keep original)."
                }),
            },
        }

    def transfer(self,
                 target: torch.Tensor,
                 reference: torch.Tensor,
                 strength: float = 1.0,
                 match_luminance: bool = True,
                 match_green_red: bool = True,
                 match_blue_yellow: bool = True,
                 mask: torch.Tensor | None = None) -> tuple:

        batch_size = target.shape[0]
        results = []

        if reference.shape[0] < batch_size:
            reference = reference.expand(batch_size, -1, -1, -1)

        for i in range(batch_size):
            tgt_bgr = tensor_to_numpy_bgr(target[i].unsqueeze(0))
            ref_bgr = tensor_to_numpy_bgr(reference[i].unsqueeze(0))

            if tgt_bgr.shape[:2] != ref_bgr.shape[:2]:
                ref_bgr = cv2.resize(ref_bgr, (tgt_bgr.shape[1], tgt_bgr.shape[0]),
                                     interpolation=cv2.INTER_AREA)

            transferred_bgr = lab_color_transfer(
                tgt_bgr, ref_bgr, strength,
                match_l=match_luminance,
                match_a=match_green_red,
                match_b=match_blue_yellow,
            )

            result_tensor = numpy_bgr_to_tensor(transferred_bgr)

            if mask is not None:
                m = mask[i] if mask.shape[0] > i else mask[0]
                m = m.unsqueeze(-1).cpu()
                if m.shape[:2] != result_tensor.shape[1:3]:
                    m_np = m.numpy()
                    m_np = cv2.resize(m_np, (result_tensor.shape[2], result_tensor.shape[1]),
                                      interpolation=cv2.INTER_LINEAR)
                    m = torch.from_numpy(m_np).unsqueeze(-1)

                original = target[i].cpu()
                result_tensor = result_tensor.squeeze(0)
                result_tensor = original * (1.0 - m) + result_tensor * m
                result_tensor = result_tensor.unsqueeze(0)

            results.append(result_tensor)

        output = torch.cat(results, dim=0)
        return (output,)


COLOR_NODE_CLASS_MAPPINGS = {
    "FG_ColorBlend": FG_ColorBlend,
    "FG_HueCorrect": FG_HueCorrect,
    "FG_ImageForensics": FG_ImageForensics,
    "FG_LABColorTransfer": FG_LABColorTransfer,
}
COLOR_NODE_DISPLAY_NAME_MAPPINGS = {
    "FG_ColorBlend": "🗑️ Color Blender",
    "FG_HueCorrect": "🗑️ Hue Correct (LAB)",
    "FG_ImageForensics": "🗑️ Image Forensics",
    "FG_LABColorTransfer": "🗑️ LAB Color Transfer",
}

