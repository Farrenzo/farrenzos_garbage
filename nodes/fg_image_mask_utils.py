"""
Loads an image, fills or erases a rectangular region,
and outputs the modified image and a mask of the affected area.

┌─────────────────────────────────────┐
│ Box Fill / Erase XY + Width Height  │
├─────────────────────────────────────┤
│ ○ images                  IMAGE  ○  │
│                           MASK   ○  │
│                                     │
│  mode          [fill|erase]         │
│  x_coordinate  [0      ]            │
│  y_coordinate  [0      ]            │
│  width         [625    ]            │
│  height        [112    ]            │
│  fill_color    [#FFFFFF]            │
└─────────────────────────────────────┘
"""
import math
import torch
import torch.nn.functional as F
import numpy as np
import comfy.utils
from PIL import Image, ImageDraw


class FG_CoordinatesBoxFill:
    
    def __init__(self):
        self.NODE_NAME = "Coordinate Box Fill"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The image to edit (B,H,W,C tensor)."}),
                "mode": (["fill", "erase"], {"default": "fill",
                          "tooltip": "fill = solid color rectangle. erase = make region transparent (alpha=0)."}),
                "x_coordinate": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1,
                                         "tooltip": "Left edge of the rectangle."}),
                "y_coordinate": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1,
                                         "tooltip": "Top edge of the rectangle."}),
                "width": ("INT", {"default": 625, "min": 0, "max": 8192, "step": 1,
                                  "tooltip": "Width of the rectangle. 0 = full image width."}),
                "height": ("INT", {"default": 112, "min": 0, "max": 8192, "step": 1,
                                   "tooltip": "Height of the rectangle. 0 = full image height."}),
                "fill_color": ("STRING", {"default": "#FFFFFF",
                                          "tooltip": "Fill color as hex (ignored in erase mode)."}),
            },
        }

    FUNCTION = "fill_image_with_color"
    CATEGORY = "Farrenzo's Garbage/Image/Utils"
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("IMAGE", "MASK",)

    def fill_image_with_color(self, images, mode, x_coordinate, y_coordinate, width, height, fill_color):
        """
        Fill or erase a rectangular region on each image in the batch.

        Modes:
            fill  - Draw a solid color rectangle. Output stays RGB (B,H,W,3).
            erase - Set the region's alpha to 0 (transparent). Output becomes RGBA (B,H,W,4).

        Width/height of 0 means "use full image dimension."

        Returns:
            - images: the modified image batch
            - masks:  a binary mask batch (B, H, W) where 1.0 = affected region
        """
        result_images = []
        result_masks = []

        # Per-frame PIL round trip, so tick once per frame.
        pbar = comfy.utils.ProgressBar(images.shape[0])

        for i in range(images.shape[0]):
            img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)

            # Determine source channels
            channels = img_np.shape[2] if len(img_np.shape) == 3 else 1
            if channels == 4:
                pil_img = Image.fromarray(img_np, mode="RGBA")
            elif channels == 3:
                pil_img = Image.fromarray(img_np, mode="RGB")
            else:
                pil_img = Image.fromarray(img_np[:, :, 0], mode="L").convert("RGB")

            img_w, img_h = pil_img.size

            # Width/height of 0 = full image
            rect_w = width if width > 0 else img_w
            rect_h = height if height > 0 else img_h

            # Clamp the rectangle to image bounds
            x1 = max(0, min(x_coordinate, img_w))
            y1 = max(0, min(y_coordinate, img_h))
            x2 = max(0, min(x_coordinate + rect_w, img_w))
            y2 = max(0, min(y_coordinate + rect_h, img_h))

            has_rect = x2 > x1 and y2 > y1

            if mode == "erase":
                # Convert to RGBA if needed, then zero out alpha in the region
                if pil_img.mode != "RGBA":
                    pil_img = pil_img.convert("RGBA")

                if has_rect:
                    pixels = np.array(pil_img)
                    pixels[y1:y2, x1:x2, 3] = 0  # alpha → 0
                    pil_img = Image.fromarray(pixels, mode="RGBA")

                # RGBA → tensor (H, W, 4)
                out_np = np.array(pil_img).astype(np.float32) / 255.0

            else:  # fill
                draw = ImageDraw.Draw(pil_img)
                if has_rect:
                    draw.rectangle((x1, y1, x2, y2), fill=fill_color)

                # Keep original channel count → tensor
                out_np = np.array(pil_img).astype(np.float32) / 255.0

                # If image was grayscale-ish and now 2D, expand
                if len(out_np.shape) == 2:
                    out_np = np.stack([out_np] * 3, axis=-1)

            result_images.append(torch.from_numpy(out_np))

            # Build mask: 1.0 inside the rect, 0.0 outside
            mask = np.zeros((img_h, img_w), dtype=np.float32)
            if has_rect:
                mask[y1:y2, x1:x2] = 1.0
            result_masks.append(torch.from_numpy(mask))
            pbar.update(1)

        out_images = torch.stack(result_images)
        out_masks = torch.stack(result_masks)

        return (out_images, out_masks,)


"""
Composites the masked region of an image batch onto a backdrop:
a solid color, transparency, or a second image. Built for rotoscoping
(subject on a flat white field) but works for any keep-the-mask job.

┌──────────────────────────────────────────┐
│ Mask Backdrop                            │
├──────────────────────────────────────────┤
│ ○ images                     IMAGE    ○  │
│ ○ mask                       MASK     ○  │
│ ○ backdrop_image (opt)                   │
│                                          │
│  backdrop      [color|transparent|image] │
│  color         [#FFFFFF]                 │
│  invert_mask   [false]                   │
│  expand        [0      ]                 │
│  feather       [0      ]                 │
└──────────────────────────────────────────┘

mask = 1 keeps the source pixel, mask = 0 shows the backdrop.
Unlike ImageCompositeMasked, batch length follows `images`, so a
single-frame backdrop will not collapse an 81-frame sequence to 1.
"""

def _hex_to_rgb(value):
    """'#RRGGBB' / 'RRGGBB' / '#RGB' -> (r, g, b) floats in 0..1."""
    s = str(value).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        s = "FFFFFF"
    try:
        r, g, b = (int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        r = g = b = 1.0
    return (r, g, b)


def _match_batch(tensor, batch):
    """Repeat / trim a tensor along dim 0 so it is `batch` long."""
    n = tensor.shape[0]
    if n == batch:
        return tensor
    if n == 1:
        return tensor.repeat(batch, *([1] * (tensor.dim() - 1)))
    if n < batch:
        reps = (batch + n - 1) // n
        tensor = tensor.repeat(reps, *([1] * (tensor.dim() - 1)))
    return tensor[:batch]


def _expand_mask(mask, pixels):
    """Dilate (positive) or erode (negative) a (B,1,H,W) mask by `pixels`."""
    if pixels == 0:
        return mask
    k = abs(int(pixels)) * 2 + 1
    pad = k // 2
    if pixels > 0:
        return F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    return -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=pad)


def _feather_mask(mask, radius):
    """Separable box blur, applied twice — cheap approximation of a gaussian."""
    if radius <= 0:
        return mask
    k = int(radius) * 2 + 1
    pad = k // 2
    kern = torch.ones(1, 1, 1, k, device=mask.device, dtype=mask.dtype) / k
    for _ in range(2):
        mask = F.pad(mask, (pad, pad, 0, 0), mode="replicate")
        mask = F.conv2d(mask, kern)
        mask = F.pad(mask, (0, 0, pad, pad), mode="replicate")
        mask = F.conv2d(mask, kern.transpose(-1, -2))
    return mask


class FG_MaskBackdrop:

    def __init__(self):
        self.NODE_NAME = "Mask Backdrop"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Source frames (B,H,W,C). Batch length of the output follows this input."}),
                "mask": ("MASK", {"tooltip": "1.0 keeps the source pixel, 0.0 shows the backdrop. Resized to the image if it does not match."}),
                "backdrop": (["color", "transparent", "image"], {"default": "color",
                             "tooltip": "color = flat fill. transparent = RGBA with alpha from the mask. image = use backdrop_image."}),
                "color": ("STRING", {"default": "#FFFFFF",
                          "tooltip": "Backdrop color as hex. Only used when backdrop = color."}),
                "invert_mask": ("BOOLEAN", {"default": False,
                                "tooltip": "Flip the mask, so the masked region becomes the backdrop instead."}),
                "expand": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1,
                           "tooltip": "Grow (+) or shrink (-) the mask in pixels. A small negative value trims background fringe off the subject's edge."}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                            "tooltip": "Soften the mask edge in pixels. Keep at 0 for hard-edged rotoscope plates."}),
                "chunk_size": ("INT", {"default": 16, "min": 1, "max": 512, "step": 1,
                               "tooltip": "Frames composited per pass. Lower it if a long full-res batch runs out of memory."}),
            },
            "optional": {
                "backdrop_image": ("IMAGE", {"tooltip": "Backdrop frames, used when backdrop = image. A single frame is reused across the batch."}),
            },
        }

    FUNCTION = "apply_backdrop"
    CATEGORY = "Farrenzo's Garbage/Image/Utils"
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("IMAGE", "MASK",)

    def apply_backdrop(self, images, mask, backdrop, color, invert_mask,
                       expand, feather, chunk_size=16, backdrop_image=None):
        device = images.device
        dtype = images.dtype
        batch, height, width = images.shape[0], images.shape[1], images.shape[2]

        # --- normalise the mask to (B,1,H,W) on the image's device ---
        m = mask
        if m.dim() == 2:
            m = m.unsqueeze(0)
        if m.dim() == 4:                      # tolerate (B,H,W,1) from sloppy upstream nodes
            m = m.squeeze(-1) if m.shape[-1] == 1 else m[..., 0]
        m = m.to(device=device, dtype=dtype).unsqueeze(1)

        # Match the batch and invert while the mask is still at its source
        # resolution — both are linear, so the result is identical to doing
        # it after the upscale, and far cheaper on a long full-res batch.
        m = _match_batch(m, batch)
        if invert_mask:
            m = 1.0 - m

        if backdrop == "image":
            if backdrop_image is None:
                raise ValueError("FG_MaskBackdrop: backdrop is set to 'image' but no backdrop_image is connected.")
            bg_src = _match_batch(backdrop_image[..., :3].to(device=device, dtype=dtype), batch)
        else:
            rgb = (0.0, 0.0, 0.0) if backdrop == "transparent" else _hex_to_rgb(color)
            bg_src = None
            flat = torch.tensor(rgb, device=device, dtype=dtype).view(1, 1, 1, 3)

        step = max(1, int(chunk_size))
        pbar = comfy.utils.ProgressBar(batch)
        out_chunks = []
        mask_chunks = []

        for start in range(0, batch, step):
            stop = min(start + step, batch)
            n = stop - start

            mc = m[start:stop]
            if mc.shape[-2:] != (height, width):
                mc = F.interpolate(mc, size=(height, width), mode="bilinear", align_corners=False)
            mc = _expand_mask(mc, expand)
            mc = _feather_mask(mc, feather)
            mc = mc.clamp(0.0, 1.0)

            alpha = mc.permute(0, 2, 3, 1)    # (n,H,W,1) for broadcasting over channels

            chunk = images[start:stop]
            src = chunk[..., :3] if chunk.shape[-1] >= 3 else chunk.repeat(1, 1, 1, 3)

            if bg_src is None:
                bg = flat
            else:
                bg = bg_src[start:stop]
                if bg.shape[1:3] != (height, width):
                    bg = F.interpolate(bg.permute(0, 3, 1, 2), size=(height, width),
                                       mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

            out = src * alpha + bg * (1.0 - alpha)

            if backdrop == "transparent":
                out = torch.cat([out, alpha], dim=-1)   # (n,H,W,4)

            out_chunks.append(out)
            mask_chunks.append(mc.squeeze(1))
            pbar.update(n)

        return (torch.cat(out_chunks).contiguous(), torch.cat(mask_chunks).contiguous(),)


"""
┌─────────────────────────────────────┐
│          Auto Crop n' Clip          │
├─────────────────────────────────────┤
│ ○ image               stitcher   ○  │
│ ○ mask           cropped_image   ○  │
│                   cropped_mask   ○  │
│                                     │
│ <SWITCH> resize to target           │
│ <INT> target width / height         │
│ <COMBO> output padding              │
│ <INT> mask blend pixels             │
│ <FLOAT> mask extend factor          │
└─────────────────────────────────────┘
Crops a region around the mask, shaped to the target aspect (square by
default), and blows it up with Lanczos-3 for sampling. The stitcher
records everything the re-stitch needs, so nothing is recomputed from
widgets later.

Stitch rules, and why they exist:
  * Every pixel inside the mask is 100% the processed image. The blend
    feathers OUTWARD from the mask edge only, so nothing you inpainted out
    (extra fingers etc.) can show through a soft edge.
  * Only the mask + its outward feather ring is pasted back. The rest of
    the crop is context for the sampler and is thrown away, so VAE colour
    drift and the slight softening of an up/down Lanczos round trip never
    land on pixels you didn't ask to change.
"""

_HARD_THRESHOLD = 0.05  # mask values above this count as "replace fully"


def _lanczos_matrix(in_size, out_size, device, dtype, a=3):
    """Dense (out_size, in_size) Lanczos-a resampling matrix.

    Downscaling stretches the kernel by the scale factor, so it low-passes
    before it decimates instead of aliasing (same weights as PIL's LANCZOS
    away from the borders). Taps past the edge clamp to the edge pixel.
    """
    scale = out_size / in_size
    stretch = max(1.0, 1.0 / scale)
    support = a * stretch
    centers = (torch.arange(out_size, dtype=torch.float64) + 0.5) / scale - 0.5
    taps = int(math.ceil(2 * support)) + 1
    left = torch.floor(centers - support).long() + 1
    idx = left[:, None] + torch.arange(taps)[None, :]
    x = (idx.double() - centers[:, None]) / stretch
    w = torch.sinc(x) * torch.sinc(x / a)
    w[x.abs() >= a] = 0.0
    w = w / w.sum(dim=1, keepdim=True)
    m = torch.zeros(out_size, in_size, dtype=torch.float64)
    m.scatter_add_(1, idx.clamp(0, in_size - 1), w)
    return m.to(device=device, dtype=dtype)


def _lanczos_resize(img, height, width):
    """(B,H,W,C) -> (B,height,width,C), separable Lanczos-3 in float32, clamped to 0..1.

    Stays on the tensor's device and never round-trips through 8-bit.
    """
    h, w = img.shape[1], img.shape[2]
    if (h, w) == (height, width):
        return img
    x = img.float()
    if h != height:
        x = torch.einsum("oh,bhwc->bowc", _lanczos_matrix(h, height, x.device, x.dtype), x)
    if w != width:
        x = torch.einsum("ow,bhwc->bhoc", _lanczos_matrix(w, width, x.device, x.dtype), x)
    return x.clamp(0.0, 1.0).to(img.dtype)


def _mask_like_image(mask, batch, height, width, device):
    """Any MASK layout -> (B,H,W) float on `device`, at the image's resolution."""
    m = mask
    if m.dim() == 2:
        m = m.unsqueeze(0)
    if m.dim() == 4:
        m = m.squeeze(-1) if m.shape[-1] == 1 else m[..., 0]
    m = _match_batch(m.to(device=device, dtype=torch.float32), batch)
    if m.shape[-2:] != (height, width):
        m = F.interpolate(m.unsqueeze(1), size=(height, width),
                          mode="bilinear", align_corners=False).squeeze(1)
    return m.clamp(0.0, 1.0)


def _crop_padded(t, left, top, cw, ch):
    """Crop (B,H,W,C) at a rect that may hang off the image; overhang is edge-replicated."""
    h, w = t.shape[1], t.shape[2]
    pl, pt = max(0, -left), max(0, -top)
    pr, pb = max(0, left + cw - w), max(0, top + ch - h)
    x = t
    if pl or pr or pt or pb:
        x = F.pad(x.permute(0, 3, 1, 2), (pl, pr, pt, pb), mode="replicate").permute(0, 2, 3, 1)
    return x[:, top + pt: top + pt + ch, left + pl: left + pl + cw, :]


def _place(center, size, limit):
    """Start coordinate for a span of `size` centred on `center`, kept inside [0, limit)."""
    if size <= limit:
        return min(max(int(round(center - size / 2)), 0), limit - size)
    return (limit - size) // 2  # bigger than the image: centre it, overhang gets padded


def _round_up(v, mult):
    return v if mult <= 1 else -(-v // mult) * mult


class FG_AutoCropnClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK", {"tooltip": "Area to inpaint. Everything inside it is fully replaced on stitch."}),

                "skip_node": ("BOOLEAN", {"default": False,
                    "tooltip": "Skip this node if you do not want a crop, or if the crop is all over the place."}),
                "output_resize_to_target_size": ("BOOLEAN", {"default": True,
                    "tooltip": "On: Lanczos-resize the crop to the target size. Off: crop at 1:1, no resampling at all."}),
                "output_target_width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 1,
                    "tooltip": "Sampling width. Width/height also set the crop's shape, even with resizing off."}),
                "output_target_height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 1}),
                "output_padding": (["0", "8", "16", "32", "64", "128", "256", "512"], {"default": "32",
                    "tooltip": "Round the output size up to a multiple of this."}),

                "mask_blend_pixels": ("INT", {"default": 16, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Feather width OUTSIDE the mask edge, in original-image pixels. The inside of the mask is never blended."}),
                "mask_extend_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Context around the mask: the crop is this many times the mask's bounding box (1.5 = 25% extra on each side)."}),
            }
        }

    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask")
    FUNCTION = "crop_me"
    CATEGORY = "Farrenzo's Garbage/Image"
    DESCRIPTION = "Crops an image around a mask for inpainting."

    def crop_me(
        self,
        image,
        mask,
        skip_node,
        output_resize_to_target_size,
        output_target_width,
        output_target_height,
        output_padding,
        mask_blend_pixels,
        mask_extend_factor
    ):
        if skip_node:
            stitcher = {
                "requires_stitch": False,
                "original": None,
                "rect": None,
                "blend": None,
                "context": None,
            }
            return (stitcher, image, mask)

        device = image.device
        batch, img_h, img_w = image.shape[0], image.shape[1], image.shape[2]
        pad_mult = int(output_padding)
        blend_px = int(mask_blend_pixels)

        m = _mask_like_image(mask, batch, img_h, img_w, device)
        hard = (m > _HARD_THRESHOLD).float()

        # One rect for the whole batch (union of every frame's mask), so a
        # sequence keeps a stable crop instead of jittering frame to frame.
        ys, xs = torch.nonzero(hard.amax(dim=0) > 0, as_tuple=True)
        if ys.numel() == 0:
            print("[FG_AutoCropnClip] mask is empty: cropping the whole image, the stitch will change nothing.")
            y0, y1, x0, x1 = 0, img_h, 0, img_w
        else:
            y0, y1 = ys.min().item(), ys.max().item() + 1
            x0, x1 = xs.min().item(), xs.max().item() + 1

        # Context: extend factor, but never less than the feather ring needs
        # (dilate + two box blurs reach ~3x blend_px past the mask edge).
        margin = 3 * blend_px + 4
        bw, bh = x1 - x0, y1 - y0
        cw = max(bw * mask_extend_factor, bw + 2 * margin)
        ch = max(bh * mask_extend_factor, bh + 2 * margin)

        aspect = output_target_width / output_target_height
        if cw / ch < aspect:
            cw = ch * aspect
        else:
            ch = cw / aspect
        cw, ch = int(math.ceil(cw)), int(math.ceil(ch))

        if output_resize_to_target_size:
            out_w = _round_up(output_target_width, pad_mult)
            out_h = _round_up(output_target_height, pad_mult)
        else:
            cw, ch = _round_up(cw, pad_mult), _round_up(ch, pad_mult)
            out_w, out_h = cw, ch

        left = _place((x0 + x1) / 2, cw, img_w)
        top = _place((y0 + y1) / 2, ch, img_h)

        crop_img = _crop_padded(image[..., :3], left, top, cw, ch)
        crop_hard = _crop_padded(hard.unsqueeze(-1), left, top, cw, ch)[..., 0]

        # Blend mask at ORIGINAL resolution: 1.0 inside the mask, feathered outward only.
        blend = crop_hard.unsqueeze(1)
        if blend_px > 0:
            ring = _feather_mask(_expand_mask(blend, blend_px), blend_px).clamp(0.0, 1.0)
            blend = torch.maximum(blend, ring)
        blend = blend.squeeze(1)

        # Pixels that are real image (not padding) and untouched by the blend:
        # the reference for colour matching on stitch.
        valid = torch.zeros(ch, cw, device=device)
        valid[max(0, -top):min(ch, img_h - top), max(0, -left):min(cw, img_w - left)] = 1.0
        context = (blend < 1e-3).float() * valid
        blend = blend * valid  # padding is never stitched

        out_img = _lanczos_resize(crop_img, out_h, out_w)
        out_mask = crop_hard
        if (out_h, out_w) != (ch, cw):
            out_mask = F.interpolate(
                crop_hard.unsqueeze(1), 
                size=(out_h, out_w),
                mode="bilinear", align_corners=False
            ).squeeze(1)
            out_mask = (out_mask > 0.5).float()

        stitcher = {
            "requires_stitch": True,
            "original": image.cpu(),
            "rect": (left, top, cw, ch),
            "blend": blend.cpu(),
            "context": context.cpu(),
        }
        return (stitcher, out_img, out_mask)


class FG_AutoCropnClipReStitcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "processed_image": ("IMAGE",),
                "color_match": ("BOOLEAN", {"default": True,
                    "tooltip": "Shift the processed crop's colour to match the original, measured on the context pixels outside the blend. Counters VAE colour drift."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    FUNCTION = "inpaint_stitch"
    CATEGORY = "Farrenzo's Garbage/Image"
    DESCRIPTION = "Stitches an image cropped with Auto Crop n' Clip back into the original image."

    @staticmethod
    def _color_match(proc, orig, context):
        """Per-frame, per-channel mean/std transfer, measured only on `context` pixels."""
        w = context.unsqueeze(-1)
        n = w.sum(dim=(1, 2), keepdim=True)
        enough = n > 64
        n = n.clamp(min=1.0)
        mu_o = (orig * w).sum(dim=(1, 2), keepdim=True) / n
        mu_p = (proc * w).sum(dim=(1, 2), keepdim=True) / n
        sd_o = (((orig - mu_o) ** 2) * w).sum(dim=(1, 2), keepdim=True).div(n).sqrt()
        sd_p = (((proc - mu_p) ** 2) * w).sum(dim=(1, 2), keepdim=True).div(n).sqrt()
        gain = (sd_o / (sd_p + 1e-6)).clamp(0.5, 2.0)
        out = ((proc - mu_p) * gain + mu_o).clamp(0.0, 1.0)
        return torch.where(enough, out, proc)

    def inpaint_stitch(self, stitcher, processed_image, color_match=True):
        if not stitcher["requires_stitch"]:
            return (processed_image,)

        device = processed_image.device
        batch = processed_image.shape[0]
        left, top, cw, ch = stitcher["rect"]

        # Batch follows processed_image (e.g. one source, four sampler variants).
        out = _match_batch(stitcher["original"].to(device), batch).clone().float()
        blend = _match_batch(stitcher["blend"].to(device), batch)
        context = _match_batch(stitcher["context"].to(device), batch)
        img_h, img_w = out.shape[1], out.shape[2]

        proc = processed_image[..., :3].float()
        ph, pw = proc.shape[1], proc.shape[2]
        if abs(pw / ph - cw / ch) > 0.02:
            print(f"[FG_AutoCropnClipReStitcher] processed image is {pw}x{ph}, crop was {cw}x{ch}: "
                  "aspect differs, the result will be stretched.")
        proc = _lanczos_resize(proc, ch, cw)  # always land back on the exact crop size

        orig_crop = _crop_padded(out[..., :3], left, top, cw, ch)
        if color_match:
            proc = self._color_match(proc, orig_crop, context)

        a = blend.unsqueeze(-1)
        merged = proc * a + orig_crop * (1.0 - a)  # a == 1.0 inside the mask: pure processed pixels

        y0, y1 = max(top, 0), min(top + ch, img_h)
        x0, x1 = max(left, 0), min(left + cw, img_w)
        out[:, y0:y1, x0:x1, :3] = merged[:, y0 - top:y1 - top, x0 - left:x1 - left, :]
        return (out,)