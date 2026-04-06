"""
Save Image (Clean) - ComfyUI Node
=================================
Saves images with clean filenames and format options.

Features:
- Single image: no counter suffix
- Batch images: 3-digit counter (000-999)
- Format choice: PNG (default) or WebP
- Optional metadata embedding
- WebP options: lossless, quality, method

┌─────────────────────────────────────┐
│ Save Image (Clean)                  │
├─────────────────────────────────────┤
│ ○  Images                           │
│ <→ TEXT INPUT_File Name>            │
│ - Auto Populated to %HMSf%          │
│                                     │
│ <▼ FORMAT> PNG / WEBP               │
│ <BOOL _ embed_metadata>             │
│                                     │
│ ── WebP Options ──                  │
│ <BOOL _ lossless>                   │
│ <INT  _ quality>                    │
│ <INT  _ method>                     │
└─────────────────────────────────────┘
"""
import os
import json
import datetime
import numpy as np
from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args
from ._fg_helperfunctions import log


class FG_SaveImage:
    """
    SaveImage variant with cleaner filenames and format options.
    - Single image: no counter suffix
    - Batch images: 3-digit counter (000-999)
    - Supports PNG and WebP formats
    """
    
    FORMATS = ["png", "webp"]
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
                    "tooltip": "Image format: PNG (lossless, larger) or WebP (configurable compression)."
                }),
                "embed_metadata": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Embed ComfyUI workflow metadata in the image."
                }),
            },
            "optional": {
                # WebP-specific options (ignored for PNG)
                "webp_lossless": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "WebP only: Use lossless compression. If False, uses lossy compression controlled by quality."
                }),
                "webp_quality": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "WebP only: Quality for lossy compression (1-100). Ignored if lossless=True."
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
    DESCRIPTION = "Saves images with clean filenames. Supports PNG and WebP formats with metadata embedding."
    COLOR = "#332922"
    BGCOLOR = "#593930"

    def save_images(
        self, 
        images, 
        filename_prefix="%HMSf%",
        format="png",
        embed_metadata=True,
        webp_lossless: bool = True,
        webp_quality=100,
        webp_method="default",
        prompt=None, 
        extra_pnginfo=None
    ):
        # Resolve the output folder
        full_output_folder, filename_base, subfolder = self._get_output_path(filename_prefix)
        
        batch_size = len(images)
        results = []
        ext = f".{format}"
        
        for batch_number, image in enumerate(images):
            # Convert tensor to PIL Image
            img = self._tensor_to_pil(image)
            
            # Determine filename
            if batch_size == 1:
                file = f"{filename_base}{ext}"
                file = self._avoid_collision(full_output_folder, filename_base, ext)
            else:
                file = f"{filename_base}_{batch_number:03d}{ext}"
            
            filepath = os.path.join(full_output_folder, file)
            
            # Save based on format
            if format == "png":
                self._save_png(img, filepath, embed_metadata, prompt, extra_pnginfo)
            else:  # webp
                self._save_webp(
                    img,
                    filepath,
                    embed_metadata,
                    prompt,
                    extra_pnginfo,
                    webp_lossless,
                    webp_quality,
                    webp_method
                )
            
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
            exif = self._create_webp_exif(img, prompt, extra_pnginfo)
            if exif:
                save_kwargs["exif"] = exif
        
        img.save(filepath, **save_kwargs)
    
    def _create_webp_exif(self, img, prompt, extra_pnginfo):
        """
        Create EXIF metadata for WebP images.
        
        WebP supports EXIF but not arbitrary text chunks like PNG.
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
    
    def _get_output_path(self, filename_prefix: str):
        """Resolve output folder and compute filename with variable substitution."""
        
        # Variable substitution
        if "%HMSf%" in filename_prefix:
            filename_prefix = filename_prefix.replace(
                "%HMSf%", 
                f"{datetime.datetime.now():%H%M%S%f}"
            )

        # Split into subfolder and filename
        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename_base = os.path.basename(os.path.normpath(filename_prefix))
        
        full_output_folder = os.path.join(self.output_dir, subfolder)

        # Security check
        def _security_check(a:str, b:str) -> bool:
            a = os.path.abspath(a.replace("\\", "/").lower())
            b = os.path.abspath(b.replace("\\", "/").lower())
            return a == b
        if not _security_check(self.output_dir, full_output_folder):
            log(f"{self.NODE_NAME}💾: Saving image outside: -> {self.output_dir}", message_type="warning")
        
        # Ensure folder exists
        os.makedirs(full_output_folder, exist_ok=True)
        
        return full_output_folder, filename_base, subfolder
    
    def _avoid_collision(self, folder: str, basename: str, ext: str) -> str:
        """
        If file exists, append a suffix. 
        Handles the rare case of microsecond collision.
        """
        filename = f"{basename}{ext}"
        filepath = os.path.join(folder, filename)
        
        if not os.path.exists(filepath):
            return filename
        
        # Collision: append counter
        counter = 1
        while True:
            filename = f"{basename}_{counter:02d}{ext}"
            filepath = os.path.join(folder, filename)
            if not os.path.exists(filepath):
                return filename
            counter += 1
            if counter > 99:
                # Fallback to full timestamp
                ts = datetime.datetime.now().strftime("%H%M%S%f")
                return f"{basename}_{ts}{ext}"

