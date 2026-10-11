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

import os
import re
import torch
import numpy as np
import folder_paths
import torch.nn.functional as F
import onnxruntime as ort_module
from ultralytics import YOLO
from PIL import Image, ImageDraw
from collections import namedtuple

from ._fg_helperfunctions import (
    log,
    tensor2pil,
    dilate_mask,
    make_crop_region,
    segs_to_combined_mask
)


MAX_RESOLUTION = 16384

folder_paths.add_model_folder_path("ultralytics_bbox", os.path.join(folder_paths.models_dir, "ultralytics", "bbox"))

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


# ─── bbox detection ───────────────────────────────────────────────────

# One detector, a few MB on the CPU, replaced when a different model is picked.
# Without it every widget change reloads the checkpoint from disk.
_detector = {}


def _bubble(paper, x0, y0, x1, y1):
    """The box plus every paper region it touches, holes filled.

    Lines of text split a balloon into bands, but every band touches the box,
    and the balloon's outline keeps the paper between screentone dots out. No
    morphological closing here: it would bridge thin outlines as readily as
    strokes. Searched only in a window around the box, so a balloon that opens
    onto the page background can't swallow the page.
    """
    H, W = paper.shape
    pad = max(x1 - x0, y1 - y0) // 2
    X0, Y0, X1, Y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)
    labels, _ = _label(paper[Y0:Y1, X0:X1])
    ids = np.unique(labels[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0])
    out = np.zeros((H, W), bool)
    out[Y0:Y1, X0:X1] = _fill_holes(np.isin(labels, ids[ids > 0]))
    out[y0:y1, x0:x1] = True
    return out


def _detect_mask(model_name, threshold, image, shape, paper):
    """image: (H, W, 3), paper: (H, W) bool. Returns the union of detections as (1, 1, H, W).

    Segmentation models draw their own outlines and ignore shape.
    """
    H, W = paper.shape
    path = folder_paths.get_full_path_or_raise("ultralytics_bbox", model_name)
    if path not in _detector:
        _detector.clear()
        _detector[path] = YOLO(path)
    pred = _detector[path](tensor2pil(image), conf=threshold, verbose=False)[0]
    boxes = pred.boxes.xyxy.round().int().tolist()
    log(f"Linework Composite: {len(boxes)} detection(s) found by {model_name}.")

    canvas = Image.new("L", (W, H))
    draw = ImageDraw.Draw(canvas)
    if pred.masks is not None:
        for poly in pred.masks.xy:
            if len(poly) > 2:
                draw.polygon(poly.flatten().tolist(), fill=255)
    elif shape == "ellipse":
        for x0, y0, x1, y1 in boxes:
            # through the box corners, so lettering in the corners isn't cut off
            dx, dy = (x1 - x0) * 0.2071, (y1 - y0) * 0.2071
            draw.ellipse([x0 - dx, y0 - dy, x1 + dx, y1 + dy], fill=255)
    elif shape == "box":
        for x0, y0, x1, y1 in boxes:
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=255)
    m = np.array(canvas) > 0
    if pred.masks is None and shape == "bubble":
        for x0, y0, x1, y1 in boxes:
            m |= _bubble(paper, x0, y0, x1, y1)
    return torch.from_numpy(m).float().view(1, 1, H, W)


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
                "restrict_to": ("MASK", {
                    "tooltip": "Composite the whole page, then paste only these regions "
                               "onto the coloured image. Several masks (e.g. "
                               "FG_BubbleTint's bubble_masks) are merged."}),
                "restrict_grow": ("INT", {
                    "default": 4, "min": 0, "max": 64,
                    "tooltip": "Grow and feather restrict_to by this many pixels so the "
                               "pasted regions blend in without a hard edge."}),
                "bbox_detect": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Find the text regions with a bbox model instead of (or "
                               "as well as) drawing restrict_to by hand."}),
                "bbox_model": (folder_paths.get_filename_list("ultralytics_bbox"), {
                    "tooltip": "Detector from models/ultralytics/bbox, run on the original."}),
                "bbox_threshold": ("FLOAT", {
                    "default": 0.3, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Detection confidence a box needs to be used. Lower finds "
                               "more text but also more false hits."}),
                "bbox_shape": (["box", "ellipse", "bubble"], {
                    "default": "box",
                    "tooltip": "Region pasted per detected box. ellipse: an oval through "
                               "the box corners. bubble: the white balloon the text sits "
                               "on (paper = L above ink_high - ink_feather), so screentone "
                               "outside it stays out. Ignored by -seg models, which use "
                               "their own outlines."}),
                "passes": ("INT", {
                    "default": 1, "min": 1, "max": 8,
                    "tooltip": "Feed the result back in as the coloured image this many "
                               "times, like chaining the node. Each pass pulls the text "
                               "closer to the original; 2-3 is usually as good as it gets."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "composite"
    CATEGORY = "Farrenzo's Garbage/Image/Utils"

    def composite(self, original, colored, mode, detail_radius, detail_strength,
                  chroma_blur, protect_ink=True, ink_low=12.0, ink_high=94.0,
                  ink_feather=6.0, restrict_to=None, restrict_grow=4,
                  bbox_detect=False, bbox_model=None, bbox_threshold=0.3,
                  bbox_shape="box", passes=1):
        B, H, W, _ = original.shape

        col = colored
        if col.shape[0] != B:
            col = col[:1].expand(B, -1, -1, -1) if col.shape[0] == 1 else col[:B]
        if col.shape[1] != H or col.shape[2] != W:
            col = F.interpolate(
                col.permute(0, 3, 1, 2), size=(H, W),
                mode="bicubic", align_corners=False, antialias=False,
            ).clamp(0, 1).permute(0, 2, 3, 1)
        col = col[..., :3]

        lab_o = rgb_to_lab(original[..., :3])
        L_o = lab_o[..., 0:1].permute(0, 3, 1, 2)
        detail = detail_strength * (L_o - _blur(L_o, detail_radius))

        keep = None
        if protect_ink:
            feather = max(ink_feather, 1e-3)
            dark = ((ink_low + feather - L_o) / feather).clamp(0, 1)
            light = ((L_o - (ink_high - feather)) / feather).clamp(0, 1)
            keep = torch.maximum(dark, light)          # 1 where ink or paper

        m = None
        if restrict_to is not None:
            m = restrict_to.reshape(-1, 1, *restrict_to.shape[-2:])
            if m.shape[0] != B:
                m = m.amax(0, keepdim=True)
            if m.shape[-2:] != (H, W):
                m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
        if bbox_detect:
            paper = (L_o[:, 0] > ink_high - ink_feather).numpy()
            boxes = torch.cat([_detect_mask(bbox_model, bbox_threshold, original[b, ..., :3], bbox_shape, paper[b]) for b in range(B)])
            m = boxes if m is None else torch.maximum(m, boxes)
        if m is not None:
            if restrict_grow > 0:
                m = F.max_pool2d(m, restrict_grow * 2 + 1, stride=1, padding=restrict_grow)
                m = _blur(m, restrict_grow)
            m = m.clamp(0, 1).permute(0, 2, 3, 1)

        # Each pass feeds its output back in as the coloured image, the same as
        # chaining the node: the model's L is blurred again and the original's
        # detail re-added, so the text converges on the original's.
        for _ in range(passes):
            lab_c = rgb_to_lab(col)
            L_c = lab_c[..., 0:1].permute(0, 3, 1, 2)

            if mode == "replace":
                L_out = L_o
            elif mode == "off":
                L_out = L_c
            else:
                # low frequency (shading) from the model, high frequency (lines,
                # text, screentone) from the original
                L_out = _blur(L_c, detail_radius) + detail

            ab = lab_c[..., 1:3].permute(0, 3, 1, 2)
            if chroma_blur > 0:
                ab = torch.cat([_blur(ab[:, 0:1], chroma_blur),
                                _blur(ab[:, 1:2], chroma_blur)], dim=1)

            if keep is not None:
                ab = ab * (1.0 - keep)
                L_out = L_out * (1.0 - keep) + L_o * keep

            L_out = L_out.clamp(0, 100)
            out = lab_to_rgb(torch.cat([L_out, ab], dim=1).permute(0, 2, 3, 1))
            if m is not None:
                # Composite the whole page, then paste the masked regions of it
                # onto the coloured image.
                out = torch.lerp(col, out, m)
            col = out

        return (out,)

"""
FG_BubbleTint — find speech bubbles, tint the lettering inside them.

Three problems here, only two of which a computer can do:

  colouring the ink   trivial. FG_LineworkComposite's protect_ink already builds
                      this exact mask to force ink neutral; tinting is the same
                      mask with the opposite operation.
  finding bubbles     doable. A bubble is a near-white closed region, containing
                      dark pixels, surrounded by darker art. Threshold, fill
                      holes, label, filter by area and solidity.
  who is speaking     not automatable. That needs reading the dialogue.

So: run once with an empty colour_map to get a numbered preview, read off which
bubble belongs to whom, then fill in "1:#2a4bd7, 2:#c2185b, 3:#2a4bd7".

Run this AFTER FG_LineworkComposite. The composite's protect_ink deliberately
forces ink to neutral, so anything you tint before it gets undone.

Tinting keeps the ink dark. Pure black has no chroma to rotate, so a hue alone
would be invisible — L has to lift a little for the colour to read at all. That
is what lift controls, and it is why the result is navy and maroon rather than
bright blue and pink.
"""

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
    lin = xyz @ _RGB_FROM_XYZ.to(lab).T
    return torch.where(lin <= 0.0031308, lin * 12.92,
                       1.055 * lin.clamp(min=0) ** (1 / 2.4) - 0.055).clamp(0, 1)


def _hex_to_lab(h):
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    rgb = torch.tensor([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
    return rgb_to_lab(rgb.view(1, 1, 1, 3))[0, 0, 0]


_ENTRY = re.compile(r"([\d\s,]+?)\s*:\s*(#?[0-9a-fA-F]{6}|#?[0-9a-fA-F]{3})\b")


def _parse_color_map(s):
    """'1:#2a4bd7, 2:#c2185b' and '1,3:#2a4bd7 | 2:#c2185b' both work.

    Matching index-run/colour pairs directly avoids the ambiguity of commas
    separating entries AND grouping indices within one entry.
    """
    out = {}
    for idx_part, col in _ENTRY.findall(s):
        col = col if col.startswith("#") else "#" + col
        for tok in re.findall(r"\d+", idx_part):
            out[int(tok)] = col
    return out


def _close(binary, r):
    """Morphological closing: bridge gaps the lettering cuts through the paper.

    Hole-filling alone only captures lettering fully enclosed by paper. When a
    line of text runs the whole width of a balloon it splits the white region
    into bands instead, and each band looks like a separate object. Closing
    first merges them back.
    """
    if r <= 0:
        return binary
    try:
        from scipy import ndimage
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk = (yy * yy + xx * xx) <= r * r
        return ndimage.binary_closing(binary, structure=disk)
    except Exception:
        out = binary
        for _ in range(r):          # dilate
            g = out.copy()
            g[1:, :] |= out[:-1, :]; g[:-1, :] |= out[1:, :]
            g[:, 1:] |= out[:, :-1]; g[:, :-1] |= out[:, 1:]
            out = g
        for _ in range(r):          # erode
            g = out.copy()
            g[1:, :] &= out[:-1, :]; g[:-1, :] &= out[1:, :]
            g[:, 1:] &= out[:, :-1]; g[:, :-1] &= out[:, 1:]
            out = g
        return out


def _fill_holes(binary):
    """Close lettering inside a bubble so the bubble reads as one solid region."""
    try:
        from scipy import ndimage
        return ndimage.binary_fill_holes(binary)
    except Exception:
        pass

    # Fallback: flood the background inward from a padded border. The padding
    # ring must be part of the floodable set or the seed dies on the first AND.
    h, w = binary.shape
    bg = np.ones((h + 2, w + 2), bool)
    bg[1:-1, 1:-1] = ~binary
    reach = np.zeros_like(bg)
    reach[0, 0] = True
    while True:
        grown = reach.copy()
        grown[1:, :] |= reach[:-1, :]
        grown[:-1, :] |= reach[1:, :]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        grown &= bg
        if np.array_equal(grown, reach):
            break
        reach = grown
    return binary | (~reach[1:-1, 1:-1] & ~binary)


def _label(binary):
    try:
        from scipy import ndimage
        lab, n = ndimage.label(binary)
        return lab, n
    except Exception:
        # Fallback: iterative max-propagation. Slower but dependency-free.
        h, w = binary.shape
        lab = np.where(binary, np.arange(h * w).reshape(h, w) + 1, 0)
        while True:
            prev = lab.copy()
            for _ in range(2):
                lab[1:, :] = np.maximum(lab[1:, :], lab[:-1, :] * binary[1:, :])
                lab[:-1, :] = np.maximum(lab[:-1, :], lab[1:, :] * binary[:-1, :])
                lab[:, 1:] = np.maximum(lab[:, 1:], lab[:, :-1] * binary[:, 1:])
                lab[:, :-1] = np.maximum(lab[:, :-1], lab[:, 1:] * binary[:, :-1])
            if np.array_equal(prev, lab):
                break
        ids = np.unique(lab[lab > 0])
        remap = {v: i + 1 for i, v in enumerate(ids)}
        out = np.zeros_like(lab)
        for v, i in remap.items():
            out[lab == v] = i
        return out, len(ids)


class FG_BubbleTint:
    """Detect speech bubbles and tint the lettering inside each one."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Run this AFTER FG_LineworkComposite."}),
                "color_map": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Leave empty on the first run to get a numbered "
                               "preview. Then e.g. '1,3:#2a4bd7 | 2:#c2185b'."}),
                "white_threshold": ("FLOAT", {
                    "default": 0.86, "min": 0.5, "max": 1.0, "step": 0.01,
                    "tooltip": "How bright a pixel must be to count as bubble paper."}),
                "ink_threshold": ("FLOAT", {
                    "default": 0.45, "min": 0.05, "max": 0.9, "step": 0.01,
                    "tooltip": "How dark a pixel must be to count as lettering."}),
                "min_area": ("FLOAT", {
                    "default": 0.0008, "min": 0.0, "max": 0.5, "step": 0.0001,
                    "tooltip": "Smallest bubble, as a fraction of the page."}),
                "max_area": ("FLOAT", {
                    "default": 0.12, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Largest bubble. Keeps the page background out."}),
                "tint_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "lift": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How far the ink's lightness moves toward the target. "
                               "0 keeps it pure black, where no hue can show. 1 makes "
                               "it as light as the swatch."}),
            },
            "optional": {
                "restrict_to": ("MASK", {
                    "tooltip": "Only search inside this region. Use the text_mask "
                               "from FG_AnimaRegionsFromJSON if you have it."}),
                "close_radius": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Bridge gaps that lettering cuts through a balloon. "
                               "0 = auto (scaled to page size). Raise if a bubble "
                               "splits into bands; lower if nearby bubbles merge."}),
                "min_solidity": ("FLOAT", {
                    "default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Filled area / bounding box area. Bubbles are blobby "
                               "and score high; thin panel gutters score low."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "preview", "bubble_masks", "info")
    FUNCTION = "tint"
    CATEGORY = "Farrenzo's Garbage/Image/Color"
    DESCRIPTION = "Find speech bubbles and colour the lettering inside them."

    def tint(self, image, color_map, white_threshold, ink_threshold, min_area,
             max_area, tint_strength, lift, restrict_to=None, min_solidity=0.55,
             close_radius=0):
        img = image[0, ..., :3]
        H, W, _ = img.shape
        lum = (img * torch.tensor([0.2126, 0.7152, 0.0722])).sum(-1)

        paper = (lum > white_threshold).numpy()
        if restrict_to is not None:
            m = restrict_to if restrict_to.ndim == 3 else restrict_to.unsqueeze(0)
            if m.shape[-2:] != (H, W):
                m = F.interpolate(m.unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)
            paper &= (m[0] > 0.5).numpy()

        r = close_radius if close_radius > 0 else max(2, min(H, W) // 160)
        solid = _fill_holes(_close(paper, r))
        labels, n = _label(solid)

        ink = (lum < ink_threshold).numpy()
        page = H * W
        bubbles, rows = [], []

        for i in range(1, n + 1):
            comp = labels == i
            area = int(comp.sum())
            frac = area / page
            if frac < min_area or frac > max_area:
                continue
            ys, xs = np.nonzero(comp)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            box = (y1 - y0 + 1) * (x1 - x0 + 1)
            solidity = area / max(box, 1)
            if solidity < min_solidity:
                continue
            inked = int((comp & ink).sum())
            if inked < 20:              # a bubble with no lettering is just paper
                continue
            bubbles.append(comp)
            rows.append((len(bubbles), frac * 100, solidity,
                         inked / max(area, 1) * 100, (int(x0), int(y0), int(x1), int(y1))))

        cmap = _parse_color_map(color_map)
        lab = rgb_to_lab(img.unsqueeze(0))[0]
        out_lab = lab.clone()
        tinted = 0

        for idx, comp in enumerate(bubbles, start=1):
            hexcol = cmap.get(idx)
            if not hexcol:
                continue
            try:
                tgt = _hex_to_lab(hexcol)
            except Exception:
                continue

            region = torch.from_numpy(comp)
            # Soft inkiness so anti-aliased glyph edges get partial tint; a hard
            # cutoff leaves a visible fringe of untouched grey around every letter.
            t = ((ink_threshold * 100 - lab[..., 0]) / max(ink_threshold * 100, 1e-3))
            w = t.clamp(0, 1)
            w = (w * w * (3 - 2 * w)) * region.float() * tint_strength

            out_lab[..., 0] = out_lab[..., 0] + (tgt[0] - lab[..., 0]) * w * lift
            out_lab[..., 1] = out_lab[..., 1] + (tgt[1] - lab[..., 1]) * w
            out_lab[..., 2] = out_lab[..., 2] + (tgt[2] - lab[..., 2]) * w
            tinted += 1

        out_lab[..., 0] = out_lab[..., 0].clamp(0, 100)
        out = lab_to_rgb(out_lab.unsqueeze(0))

        # numbered preview
        pil = Image.fromarray((out[0].numpy() * 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil)
        for (num, _, _, _, (x0, y0, x1, y1)) in rows:
            col = cmap.get(num, "#00c000")
            draw.rectangle([x0, y0, x1, y1], outline=col, width=max(2, H // 400))
            draw.text((x0 + 4, y0 + 4), str(num), fill=col)
        preview = torch.from_numpy(np.asarray(pil).astype(np.float32) / 255).unsqueeze(0)

        if bubbles:
            masks = torch.stack([torch.from_numpy(c).float() for c in bubbles])
        else:
            masks = torch.zeros(1, H, W)

        lines = [f"page {W}x{H}, close_radius {r}, {n} white regions found, "
                 f"{len(bubbles)} kept as bubbles",
                 f"tinted {tinted} of {len(bubbles)}"]
        info_line = ""
        for (num, frac, sol, inkpct, box) in rows:
            lines.append(f"  [{num}] area {frac:5.2f}%  solidity {sol:.2f}  "
                         f"ink {inkpct:4.1f}%  bbox {box}  "
                         f"{cmap.get(num, '(no colour assigned)')}")
            if not cmap:
                info_line += f"{num}, "
        if not cmap:
            info_line += ":#000000 | "
            lines += ["", "No colour_map given — image unchanged. Look at 'preview' "
                          "for the numbering, then set e.g. '1,3:#2a4bd7 | 2:#c2185b'.", info_line]
            
        if not bubbles:
            lines += ["", "Nothing detected. Lower white_threshold if bubbles are "
                          "off-white, raise max_area if bubbles are large, lower "
                          "min_solidity if they are irregular/jagged, or raise "
                          "close_radius if lettering is splitting them into bands."]

        info = "\n".join(lines)
        print("[FG_BubbleTint]\n" + info)
        return (out, preview, masks, info)


"""
FG_ColorPicker / FG_ColorMap — colour swatch widgets that actually work in ComfyUI.

Any STRING widget whose name is "color"/"colour" or ends in "_color", "_colour"
or "_hex" gets upgraded by web/js/fg_color_swatch.js into a clickable swatch that
opens the OS colour dialog. The Python side stays a plain STRING, so the value
serialises, undoes, and round-trips through the API exactly as before — the JS
only changes how it is drawn and clicked.

Add to your pack's __init__.py:

    WEB_DIRECTORY = "./web"

and make sure it is exported alongside NODE_CLASS_MAPPINGS.
"""

_XYZ_FROM_RGB = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                              [0.2126729, 0.7151522, 0.0721750],
                              [0.0193339, 0.1191920, 0.9503041]])
_WHITE = torch.tensor([0.95047, 1.00000, 1.08883])
_EPS, _KAPPA = 216 / 24389, 24389 / 27


def _hex_to_rgb(h, fallback=(0, 0, 0)):
    h = str(h).strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return fallback
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return fallback


def _rgb_to_lab(r, g, b):
    c = torch.tensor([r / 255.0, g / 255.0, b / 255.0])
    lin = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).clamp(min=0) ** 2.4)
    xyz = (_XYZ_FROM_RGB @ lin) / _WHITE
    f = torch.where(xyz > _EPS, xyz.clamp(min=1e-8) ** (1 / 3), (_KAPPA * xyz + 16) / 116)
    L = 116 * f[1] - 16
    a = 500 * (f[0] - f[1])
    bb = 200 * (f[1] - f[2])
    C = float((a * a + bb * bb) ** 0.5)
    hue = float(torch.rad2deg(torch.atan2(bb, a)) % 360)
    return float(L), float(a), float(bb), C, hue


class FG_ColorPicker:
    """One swatch. Outputs the hex plus the numbers other nodes want."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color": ("STRING", {
                    "default": "#fe62e2",
                    "tooltip": "Click the swatch to open the OS colour dialog."}),
            },
            "optional": {
                "swatch_size": ("INT", {"default": 128, "min": 16, "max": 1024, "step": 16}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "INT", "INT", "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("hex", "swatch", "R", "G", "B", "lab_hue", "lab_chroma", "info")
    FUNCTION = "pick"
    CATEGORY = "Farrenzo's Garbage/Image/Color"
    DESCRIPTION = "Colour swatch widget. Outputs hex, RGB and LAB hue/chroma."

    def pick(self, color, swatch_size=128):
        r, g, b = _hex_to_rgb(color)
        hexs = f"#{r:02x}{g:02x}{b:02x}"
        L, a, bb, C, hue = _rgb_to_lab(r, g, b)

        img = torch.tensor([r / 255.0, g / 255.0, b / 255.0])
        swatch = img.view(1, 1, 1, 3).repeat(1, swatch_size, swatch_size, 1)

        info = (f"{hexs}  rgb({r},{g},{b})\n"
                f"LAB L={L:.1f} a={a:.1f} b={bb:.1f}\n"
                f"hue {hue:.1f} deg, chroma {C:.1f}")
        return (hexs, swatch, r, g, b, hue, C, info)


class FG_ColorMap:
    """Build FG_BubbleTint's colour_map string from swatches instead of typing hex."""

    SLOTS = 6

    @classmethod
    def INPUT_TYPES(cls):
        req = {}
        defaults = ["#1f48ea", "#fe62e2", "#2e7d32", "#ef6c00", "#6a1b9a", "#00838f"]
        for i in range(1, cls.SLOTS + 1):
            req[f"slot{i}_color"] = ("STRING", {"default": defaults[i - 1]})
            req[f"slot{i}_bubbles"] = ("STRING", {
                "default": "" if i > 2 else ("1" if i == 1 else "2"),
                "tooltip": "Bubble numbers for this colour, comma separated. "
                           "Read them off FG_BubbleTint's preview output. "
                           "Leave empty to skip the slot."})
        return {"required": req}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("color_map", "info")
    FUNCTION = "build"
    CATEGORY = "Farrenzo's Garbage/Image/Color"
    DESCRIPTION = "Assemble a colour_map string for FG_BubbleTint."

    def build(self, **kw):
        parts, lines = [], []
        for i in range(1, self.SLOTS + 1):
            idxs = str(kw.get(f"slot{i}_bubbles", "")).strip()
            if not idxs:
                continue
            nums = [t for t in "".join(
                c if c.isdigit() else " " for c in idxs).split() if t]
            if not nums:
                continue
            r, g, b = _hex_to_rgb(kw.get(f"slot{i}_color", "#000000"))
            hexs = f"#{r:02x}{g:02x}{b:02x}"
            parts.append(f"{','.join(nums)}:{hexs}")
            lines.append(f"  slot {i}: bubbles {', '.join(nums)} -> {hexs}")

        cmap = " | ".join(parts)
        info = ("color_map: " + (cmap if cmap else "(empty — fill in at least one "
                                                 "slot's bubble numbers)")
                + ("\n" + "\n".join(lines) if lines else ""))
        print("[FG_ColorMap] " + info.replace("\n", " "))
        return (cmap, info)



SEG = namedtuple(
    "SEG",
    ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
    defaults=[None]
)

class ONNXDetector:
    onnx_model = None

    def __init__(self, onnx_model):
        self.onnx_model = onnx_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)

        def _onnx_inference(image, onnx_model):
            # prepare image
            pil = tensor2pil(image)
            image = np.ascontiguousarray(pil)
            image = image[:, :, ::-1]  # to BGR image
            image = image.astype(np.float32)
            image -= [103.939, 116.779, 123.68]  # 'caffe' mode image preprocessing

            # do detection
            onnx_model = ort_module.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
            outputs = onnx_model.run(
                [s_i.name for s_i in onnx_model.get_outputs()],
                {onnx_model.get_inputs()[0].name: np.expand_dims(image, axis=0)},
            )

            labels = [op for op in outputs if op.dtype == "int32"][0]
            scores = [op for op in outputs if isinstance(op[0][0], np.float32)][0]
            boxes = [op for op in outputs if isinstance(op[0][0], np.ndarray)][0]

            # filter-out useless item
            idx = np.where(labels[0] == -1)[0][0]

            labels = labels[0][:idx]
            scores = scores[0][:idx]
            boxes = boxes[0][:idx].astype(np.uint32)

            return labels, scores, boxes

        try:
            h = image.shape[1]
            w = image.shape[2]
            labels, scores, boxes = _onnx_inference(image, self.onnx_model)

            # collect feasible item
            result = []

            for i in range(len(labels)):
                if scores[i] > threshold:
                    item_bbox = boxes[i]
                    x1, y1, x2, y2 = item_bbox

                    if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                        crop_region = make_crop_region(w, h, item_bbox, crop_factor)

                        if detailer_hook is not None:
                            crop_region = item_bbox.post_crop_region(w, h, item_bbox, crop_region)

                        crop_x1, crop_y1, crop_x2, crop_y2, = crop_region

                        # prepare cropped mask
                        cropped_mask = np.zeros((crop_y2 - crop_y1, crop_x2 - crop_x1))
                        cropped_mask[y1 - crop_y1:y2 - crop_y1, x1 - crop_x1:x2 - crop_x1] = 1
                        cropped_mask = dilate_mask(cropped_mask, dilation)

                        # make items. just convert the integer label to a string
                        item = SEG(None, cropped_mask, scores[i], crop_region, item_bbox, str(labels[i]), None)
                        result.append(item)

            shape = h, w
            segs = shape, result

            if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
                segs = detailer_hook.post_detection(segs)

            return segs
        except Exception as e:
            log(f"ONNXDetector: unable to execute.\n{e}", message="error")

    def detect_combined(self, image, threshold, dilation):
        return segs_to_combined_mask(self.detect(image, threshold, dilation, 1))

    def setAux(self, x):
        pass



class ONNXDetectorProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_name": (folder_paths.get_filename_list("onnx"), )}}

    RETURN_TYPES = ("BBOX_DETECTOR", )
    FUNCTION = "load_onnx"

    CATEGORY = "ImpactPack"

    def load_onnx(self, model_name):
        model = folder_paths.get_full_path("onnx", model_name)
        return (ONNXDetector(model), )


