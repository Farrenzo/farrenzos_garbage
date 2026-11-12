"""
Save Image - ComfyUI Node
=================================
Saves images with clean filenames and format options.

Features:
- Single image: no counter suffix
- Batch images: 3-digit counter (000-999)
- Format choice: PNG (default), JPG or WebP
- Optional metadata embedding
- Options: quality_optimized (WebP lossless / JPG optimize), quality, WebP method

┌─────────────────────────────────────┐
│ Save Image                          │
├─────────────────────────────────────┤
│ ○  Images                           │
│ <→ TEXT INPUT_File Name>            │
│ - Auto Populated to %HMSf%          │
│                                     │
│ <▼ FORMAT> PNG / JPG / WEBP         │
│ <BOOL _ embed_metadata>             │
│                                     │
│ ── JPG / WebP Options ──            │
│ <BOOL _ quality_optimized>          │
│ <INT  _ quality>                    │
│ <INT  _ method>                     │
└─────────────────────────────────────┘
"""
import os
import json
import numpy as np
from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args
from ._fg_helperfunctions import avoid_naming_collisions, get_output_path, log


class FG_SaveImage:
    """
    SaveImage variant with cleaner filenames and format options.
    - Single image: no counter suffix
    - Batch images: 3-digit counter (000-999)
    - Supports PNG, JPG and WebP formats
    """
    
    FORMATS = ["png", "jpg", "webp"]
    WEBP_METHODS = ["default", "0", "1", "2", "3", "4", "5", "6"]
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4  # PNG compression level
        self.NODE_NAME = "Save Image"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {
                    "default": "%HMSf%", 
                    "tooltip": "The prefix for the file to save. Supports %HMSf% for timestamp."
                }),
                "format": (s.FORMATS, {
                    "default": "png",
                    "tooltip": "Image format: PNG (lossless, larger), JPG (lossy, no transparency) or WebP (configurable compression)."
                }),
                "embed_metadata": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Embed ComfyUI workflow metadata in the image."
                }),
            },
            "optional": {
                # JPG / WebP options (ignored for PNG)
                "quality_optimized": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "WebP: lossless compression. JPG: optimized Huffman tables and 4:4:4 chroma (no subsampling)."
                }),
                "quality": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "JPG / WebP: Quality for lossy compression (1-100). Ignored for WebP when quality_optimized=True."
                }),
                "webp_method": (s.WEBP_METHODS, {
                    "default": "default",
                    "tooltip": "WebP only: Compression method (0=fast, 6=slowest/best). 'default' uses PIL's default."
                }),
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Farrenzo's Garbage/Image"
    DESCRIPTION = "Saves images with clean filenames. Supports PNG, JPG and WebP formats with metadata embedding."
    COLOR = "#332922"
    BGCOLOR = "#593930"

    def save_images(
        self, 
        images, 
        filename_prefix="%HMSf%",
        format="png",
        embed_metadata=True,
        quality_optimized: bool = True,
        quality=100,
        webp_method="default",
        prompt=None, 
        extra_pnginfo=None
    ):
        # Resolve the output folder
        full_output_folder, filename_base, subfolder = get_output_path(
            node_name       = self.NODE_NAME, 
            filename_prefix = filename_prefix, 
            output_path     = self.output_dir
        )
        
        batch_size = len(images)
        results = []
        ext = f".{format}"

        # Pick one base for the whole batch so no _000.._NNN file overwrites an existing one.
        if batch_size > 1:
            base, n = filename_base, 0
            while any(os.path.exists(os.path.join(full_output_folder, f"{filename_base}_{i:03d}{ext}")) for i in range(batch_size)):
                n += 1
                filename_base = f"{base}_{n:02d}"

        for batch_number, image in enumerate(images):
            # Convert tensor to PIL Image
            img = self._tensor_to_pil(image)

            # Determine filename
            if batch_size == 1:
                file = avoid_naming_collisions(full_output_folder, filename_base, ext)
            else:
                file = f"{filename_base}_{batch_number:03d}{ext}"
            
            filepath = os.path.join(full_output_folder, file)
            
            # Save based on format
            if format == "png":
                self._save_png(
                    img, 
                    filepath,
                    embed_metadata,
                    prompt,
                    extra_pnginfo
                )
            elif format == "jpg":
                self._save_jpg(
                    img, 
                    filepath, 
                    embed_metadata, 
                    prompt, 
                    extra_pnginfo, 
                    quality_optimized, 
                    quality
                )
            elif format == "webp":
                self._save_webp(
                    img,
                    filepath,
                    embed_metadata,
                    prompt,
                    extra_pnginfo,
                    quality_optimized,
                    quality,
                    webp_method
                )
            else:
                raise ValueError(f"{format} is not supported.")
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
        
        return {"ui": {"images": results}}
    
    def _tensor_to_pil(self, image_tensor) -> PILImage.Image:
        """Convert a torch tensor to PIL Image."""
        i = 255.0 * image_tensor.cpu().numpy()
        return PILImage.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    
    def _save_png(self, img: PILImage, filepath: str, embed_metadata, prompt, extra_pnginfo):
        """Save image as PNG with optional metadata."""
        metadata = None
        if embed_metadata and not args.disable_metadata:
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for key in extra_pnginfo:
                    metadata.add_text(key, json.dumps(extra_pnginfo[key]))
        
        img.save(filepath, pnginfo=metadata, compress_level=self.compress_level)
    
    def _save_jpg(self, img: PILImage, filepath: str, embed_metadata, prompt, extra_pnginfo, optimized, quality):
        """Save image as JPG with optional EXIF metadata. Alpha is flattened onto white."""
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            img = PILImage.new("RGB", rgba.size, (255, 255, 255))
            img.paste(rgba, mask=rgba.getchannel("A"))
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        save_kwargs = {"quality": quality}
        if optimized:
            save_kwargs["optimize"] = True
            save_kwargs["subsampling"] = 0

        if embed_metadata and not args.disable_metadata:
            exif = self._create_exif(img, prompt, extra_pnginfo)
            # JPEG stores EXIF in a single APP1 segment capped at 64KB.
            if exif and len(exif.tobytes()) > 65533:
                log("Workflow metadata exceeds the 64KB JPG EXIF limit, saving without metadata.", "warning")
            elif exif:
                save_kwargs["exif"] = exif

        img.save(filepath, format="JPEG", **save_kwargs)

    def _save_webp(
        self,
        img: PILImage,
        filepath: str,
        embed_metadata: bool,
        prompt,
        extra_pnginfo,
        lossless,
        quality,
        method
    ):
        """Save image as WebP with optional EXIF metadata."""
        save_kwargs = {
            "lossless": lossless,
            "quality": quality,
        }
        
        # Parse method
        if method != "default":
            save_kwargs["method"] = int(method)
        
        # Build EXIF metadata for WebP
        if embed_metadata and not args.disable_metadata:
            exif = self._create_exif(img, prompt, extra_pnginfo)
            if exif:
                save_kwargs["exif"] = exif
        
        img.save(filepath, **save_kwargs)
    
    def _create_exif(self, img, prompt, extra_pnginfo):
        """
        Create EXIF metadata for WebP and JPG images.
        
        WebP and JPG support EXIF but not arbitrary text chunks like PNG.
        We encode workflow data into EXIF fields:
        - 0x0110 (Model): prompt data
        - 0x010F (Make) and below: extra_pnginfo entries
        """
        exif_data = img.getexif()
        
        if prompt is not None:
            # Use "Model" EXIF tag for prompt
            exif_data[0x0110] = f"prompt:{json.dumps(prompt)}"
        
        if extra_pnginfo is not None:
            # Use "Make" and nearby tags for extra info
            tag = 0x010F  # Make
            for key, value in extra_pnginfo.items():
                exif_data[tag] = f"{key}:{json.dumps(value)}"
                tag -= 1
                if tag < 0x0100:  # Don't go too low in tag numbers
                    break
        
        return exif_data if exif_data else None
