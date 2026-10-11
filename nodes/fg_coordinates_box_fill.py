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
