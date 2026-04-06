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
│                                     │
│                                     │
│                                     │
└─────────────────────────────────────┘

"""

import os
import torch
import hashlib
import numpy as np
import folder_paths
import node_helpers
from PIL import Image, ImageOps, ImageSequence


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
                    "image": (sorted(files), {"image_upload": True, "tooltip": "The image you want uploaded."})
                },
        }


    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("Image", "Mask", "Width", "Height")
    FUNCTION = "load_image"
    CATEGORY = "Farrenzo's Garbage/Utils"
    DESCRIPTION = "Load an image from disk, but also get the width and height for it as well."

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
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

        return (output_image, output_mask, w, h)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"

        return True
