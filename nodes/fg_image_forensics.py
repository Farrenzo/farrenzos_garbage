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

import torch
import torch.nn.functional as F


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


