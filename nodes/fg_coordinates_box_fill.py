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
import numpy as np
from PIL import Image, ImageDraw


class CoordinatesBoxFill:
    
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
    CATEGORY = "Farrenzo's Garbage/Utils"
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

        out_images = torch.stack(result_images)
        out_masks = torch.stack(result_masks)

        return (out_images, out_masks,)