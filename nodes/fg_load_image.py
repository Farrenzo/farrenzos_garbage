"""
Loads an image, outputs the image, image mask, width & height.

┌─────────────────────────────────────┐
│      Load Image + W x H             │
├─────────────────────────────────────┤
│                           Image  ○  │
│                            Mask  ○  │
│                                     │
│                           Width  ○  │
│                          Height  ○  │
│ <→Label _image_name_>               │
│ <BUTTON choose file to load>        │
│ <INT _ crop_width>   0 = free drag  │
│ <INT _ crop_height>  0 = free drag  │
│ <CROP EDITOR — drag to select>      │
└─────────────────────────────────────┘

"""

import os
import json
import torch
import hashlib
import numpy as np
import folder_paths
import node_helpers
from PIL import Image, ImageOps, ImageSequence


def _parse_crop(crop, width, height, crop_width=0, crop_height=0):
    """Return (x0, y0, x1, y1) pixel box, or None for full image."""
    if not crop:
        return None
    try:
        data = json.loads(crop)
        x = float(data["x"])
        y = float(data["y"])
        w = float(data["w"])
        h = float(data["h"])
    except (ValueError, KeyError, TypeError):
        return None

    # Exact pixel box, written by the editor whenever a size was typed in.
    # Preferred over the normalized rect because a 4-decimal fraction of the
    # image width does not reliably round back to the number the user entered
    # -- ask for 1024 and you can get 1023, which is exactly the off-by-one
    # that makes a latent the wrong shape downstream.
    px = data.get("px")
    if isinstance(px, (list, tuple)) and len(px) == 4:
        try:
            x0, y0, x1, y1 = (int(round(float(v))) for v in px)
        except (ValueError, TypeError):
            pass
        else:
            x0 = max(0, min(width - 1, x0))
            y0 = max(0, min(height - 1, y0))
            x1 = max(x0 + 1, min(width, x1))
            y1 = max(y0 + 1, min(height, y1))
            if x0 == 0 and y0 == 0 and x1 == width and y1 == height:
                return None
            return (x0, y0, x1, y1)

    x0 = max(0, min(width - 1, round(x * width)))
    y0 = max(0, min(height - 1, round(y * height)))
    x1 = max(x0 + 1, min(width, round((x + w) * width)))
    y1 = max(y0 + 1, min(height, round((y + h) * height)))

    # No px box (stale JS, or a hand-edited crop string) but a size was typed:
    # honour the typed size, centred on whatever was dragged.
    cw = int(crop_width or 0)
    ch = int(crop_height or 0)
    if cw > 0 or ch > 0:
        pw = min(width, cw if cw > 0 else x1 - x0)
        ph = min(height, ch if ch > 0 else y1 - y0)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        x0 = int(max(0, min(width - pw, round(cx - pw / 2.0))))
        y0 = int(max(0, min(height - ph, round(cy - ph / 2.0))))
        x1, y1 = x0 + pw, y0 + ph

    if x0 == 0 and y0 == 0 and x1 == width and y1 == height:
        return None
    return (x0, y0, x1, y1)


class FG_LoadImage:
    def __init__(self):
        self.NODE_NAME = "Load Image"

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                    "image": (sorted(files), {"image_upload": True, "tooltip": "The image you want uploaded."}),
                    "crop": ("STRING", {"default": "", "tooltip": "Managed by the crop editor on the node — no need to edit by hand.",}),
                },
            "optional": {
                    "crop_width": ("INT", {
                        "default": 0, "min": 0, "max": 16384, "step": 1,
                        "tooltip": "Exact crop width in pixels. 0 = drag freely. "
                                   "With both width and height set the selection "
                                   "becomes a fixed-size box you click to place — "
                                   "set both back to 0 to clear it."}),
                    "crop_height": ("INT", {
                        "default": 0, "min": 0, "max": 16384, "step": 1,
                        "tooltip": "Exact crop height in pixels. 0 = drag freely. "
                                   "Lock one axis only and the other stays draggable, "
                                   "snapping to the locked value on release."}),
                },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("Image", "Mask", "Width", "Height")
    FUNCTION = "load_image"
    CATEGORY = "Farrenzo's Garbage/Image"
    DESCRIPTION = "Load an image from disk, but also get the width and height for it as well."

    @staticmethod
    def _resolve_path(image):
        if os.path.isabs(image):
            return image
        path = folder_paths.get_annotated_filepath(image)
        if os.path.isfile(path):
            return path
        # Belt and braces for clipspace: a mask saved with a malformed
        # original_ref can land in the input root rather than input/clipspace
        # (or the other way round). The basename is unique either way, so
        # check the other spot before giving up.
        base = os.path.basename(image.split(" [")[0])
        input_dir = folder_paths.get_input_directory()
        for candidate in (os.path.join(input_dir, "clipspace", base),
                          os.path.join(input_dir, base)):
            if os.path.isfile(candidate):
                return candidate
        return path

    def load_image(self, image, crop="", crop_width=0, crop_height=0):
        image_path = self._resolve_path(image)
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks  = []
        w, h = None, None

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))

            image_rgb = i.convert("RGB")
            if len(output_images) == 0:
                w, h = image_rgb.size
            if image_rgb.size != (w, h):
                continue

            image_np = np.array(image_rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]

            if 'A' in i.getbands():
                mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask_np)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask_np = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask_np)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32)

            output_images.append(image_tensor)
            output_masks.append(mask.unsqueeze(0))

            if img.format == "MPO":
                break

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # Outside the branch above: the crop used to sit in the single-image
        # arm only, so animated GIFs and multi-page TIFFs came out uncropped.
        src_w, src_h = w, h
        box = _parse_crop(crop, output_image.shape[2], output_image.shape[1],
                          crop_width, crop_height)
        if box is not None:
            x0, y0, x1, y1 = box
            output_image = output_image[:, y0:y1, x0:x1, :]
            # With no alpha channel the mask above is a 64x64 placeholder that
            # has no relationship to image coordinates -- slicing it with them
            # produces garbage, so only crop a mask that really came from one.
            if output_mask.shape[1] == src_h and output_mask.shape[2] == src_w:
                output_mask = output_mask[:, y0:y1, x0:x1]
            # Width/Height describe the IMAGE output, so they have to follow
            # the crop. Anything sizing a latent off them was getting the
            # uncropped file size before.
            w, h = x1 - x0, y1 - y0

        return (output_image, output_mask, w, h)

    @classmethod
    def IS_CHANGED(s, image, crop="", crop_width=0, crop_height=0):
        image_path = s._resolve_path(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        # The crop is part of the output, so it belongs in the cache key.
        # Hashing the file bytes alone meant adjusting the selection and
        # re-queueing replayed the cached tensors -- the node looked like it
        # was ignoring you.
        m.update(f"|{crop}|{crop_width}|{crop_height}".encode("utf-8"))
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image, crop="", crop_width=0, crop_height=0):
        # Same resolution load_image uses, so validation can't reject a
        # clipspace file the loader would have found.
        if not os.path.isfile(s._resolve_path(image)):
            return f"Invalid image file: {image}"
        return True