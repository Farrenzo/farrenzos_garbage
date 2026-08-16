"""
Some commonly re-used functions.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gc
import math
import json
import torch
import datetime
import numpy as np
from typing import List
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import comfy.model_management
from server import PromptServer

SCALING_METHODS = {
    "box"     : Image.BOX,
    "bicubic" : Image.BICUBIC,
    "hamming" : Image.HAMMING,
    "lanczos" : Image.LANCZOS,
    "nearest" : Image.NEAREST,
    "bilinear": Image.BILINEAR,
}

_BACKEND_NAMES = ("cuda", "xpu", "mps", "npu", "mtia", "hpu")
_ACCELERATORS = None

with open(Path(__file__).parent / "../web/js/_fg_settings.js", encoding='utf-8') as settings:
    settings_data = settings.read()
    # Really hacky way of ommitting first and last part of javascript file.
    global_settings: dict = json.loads(settings_data[settings_data.index("{"):settings_data.rindex("}") + 1])

MODEL_TYPES = global_settings["model_types"]
ASPECT_RATIOS = global_settings["aspect_ratios"]
TERMINAL_COLOR_CODES = global_settings["terminal_color_codes"]

def log(message:str, message_type:str="info") -> None:
    message_types ={
        "info"   :TERMINAL_COLOR_CODES["bold_blue"],
        "finish" :TERMINAL_COLOR_CODES["green"],
        "warning":TERMINAL_COLOR_CODES["bold_yellow"],
        "error"  :TERMINAL_COLOR_CODES["boldbackred"],
        **TERMINAL_COLOR_CODES
    }
    if message_type not in message_types.keys():
        print(f"\033[46m [🗑️ Garbãƶe] -> {message}\033[m")
    else:
        print(f"\033{message_types[message_type]} [🗑️ Garbãƶe] -> {message}\033[m")
    return


def generate_latent_image_data(
    width,
    height,
    batch_size = 1,
    model_type = "SDXL",
    vae        = None,
    image      = None,
    mask       = None,
    mask_growth_val = 6,
    device = comfy.model_management.intermediate_device()
):
    """Return a latent"""
    model_info = MODEL_TYPES[model_type]
    if vae is None:
        latent = {
            "samples": torch.zeros(
                [
                    batch_size,
                    model_info["channels"],
                    height // model_info["spatial_div"],
                    width  // model_info["spatial_div"]
                ],
                device=device,
                dtype=comfy.model_management.intermediate_dtype()
            )
        }
        latent_info = "empty"
    elif vae is not None and mask is None:
        latent = {"samples":vae.encode(image)}
        latent_info = "image"
    elif vae is not None and mask is not None:
        latent = vae_encode_inpainter(vae, image, mask, grow_mask_by=mask_growth_val)
        latent_info = "inpaint"
    return latent_info, latent


def vae_encode_inpainter(vae, pixels, mask, grow_mask_by=6):
    downscale_ratio = vae.spacial_compression_encode()
    x = (pixels.shape[1] // downscale_ratio) * downscale_ratio
    y = (pixels.shape[2] // downscale_ratio) * downscale_ratio
    mask = torch.nn.functional.interpolate(mask.reshape(
        (-1, 1, mask.shape[-2], mask.shape[-1])),
        size=(pixels.shape[1], pixels.shape[2]),
        mode="bilinear"
    )

    pixels = pixels.clone()
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % downscale_ratio) // 2
        y_offset = (pixels.shape[2] % downscale_ratio) // 2
        pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
        mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

    #grow mask by a few pixels to keep things seamless in latent space
    if grow_mask_by == 0:
        mask_erosion = mask
    else:
        kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))
        padding = math.ceil((grow_mask_by - 1) / 2)

        mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)

    m = (1.0 - mask.round()).squeeze(1)
    for i in range(3):
        pixels[:,:,:,i] -= 0.5
        pixels[:,:,:,i] *= m
        pixels[:,:,:,i] += 0.5
    t = vae.encode(pixels)

    return {"samples":t, "noise_mask": (mask_erosion[:,:,:x,:y].round())}

def unpack_images(images: list):
    unpacked_images = []
    for image in images:
        unpacked_images += [torch.unsqueeze(image, 0)]
    width, height = tensor2pil(unpacked_images[0]).size

    return unpacked_images, width, height

def unpack_masks(masks: list):
    unpacked_masks = []
    mask_width, mask_height = 0, 0
    if masks.dim() == 2:
        masks = torch.unsqueeze(masks, 0)
    for mask in masks:
        ma = torch.unsqueeze(mask, 0)
        if not is_valid_mask(ma) and ma.shape==torch.Size([1,64,64]):
            break
        else:
            unpacked_masks += [ma]
            mask_width, mask_height = tensor2pil(ma).size
    return unpacked_masks, mask_width, mask_height

# ---------------------------------------- #
# OLD
# ---------------------------------------- #
"""
def clear_memory(purge_cache: bool = False, purge_models: bool = False):
    if purge_cache:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f"cuda:{i}")
                comfy.model_management.free_memory(
                    comfy.model_management.get_total_memory(device) * 0.8,
                    device
                )
                with torch.cuda.device(i):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
    if purge_models:
        comfy.model_management.unload_all_models()
    log(f"👝 Memory purged.")
"""
# ----------------------------------------
# NEW
# ----------------------------------------
# Probed once, then cached. Order = preference when several are present.

def _is_available(name, mod):
    """torch.<backend>.is_available() isn't on every backend/version."""
    fn = getattr(mod, "is_available", None)
    if fn is None:
        fn = getattr(getattr(torch.backends, name, None), "is_available", None)
    try:
        return bool(fn()) if fn else False
    except Exception:
        return False


def get_accelerators(refresh: bool = False):
    """
    All usable torch accelerator backends as [(name, module), ...].

    Returns a list, not a single winner — a FrankenWheel-style build can expose
    cuda and xpu at the same time, and both want purging. Empty list = CPU only.
    Note: ROCm reports itself as 'cuda'.
    """
    global _ACCELERATORS
    if _ACCELERATORS is None or refresh:
        found = []
        for name in _BACKEND_NAMES:
            mod = getattr(torch, name, None)
            if mod is not None and _is_available(name, mod):
                found.append((name, mod))
        _ACCELERATORS = found
    return _ACCELERATORS


def get_devices(name, mod):
    """torch.device objects for one backend. mps is single-device, unindexed."""
    count_fn = getattr(mod, "device_count", None)
    if count_fn is None:
        return [torch.device(name)]
    try:
        count = int(count_fn())
    except Exception:
        count = 0
    return [torch.device(f"{name}:{i}") for i in range(count)]


def purge_backend(name, mod, device):
    """Best-effort cache drop. Every call is optional on some backend."""
    index = device.index

    ctx = getattr(mod, "device", None)
    handle = ctx(index) if (ctx is not None and index is not None) else None

    try:
        if handle is not None:
            handle.__enter__()
        for call in ("synchronize", "empty_cache", "ipc_collect"):
            fn = getattr(mod, call, None)
            if fn is None:
                continue
            try:
                fn()
            except Exception as e:
                log(f"⚠️ torch.{name}.{call}() failed on {device}: {e}")
    finally:
        if handle is not None:
            handle.__exit__(None, None, None)


def clear_memory(purge_cache: bool = False, purge_models: bool = False, keep: float = 0.2):
    """
    keep: fraction of total VRAM to leave loaded (0.2 == the old 0.8 free target).
    """
    if purge_cache:
        gc.collect()
        for name, mod in get_accelerators():
            for device in get_devices(name, mod):
                try:
                    comfy.model_management.free_memory(
                        comfy.model_management.get_total_memory(device) * (1.0 - keep),
                        device,
                    )
                except Exception as e:
                    log(f"⚠️ free_memory failed on {device}: {e}")
                purge_backend(name, mod, device)

    if purge_models:
        comfy.model_management.unload_all_models()

    log(f"👝 Memory purged.")
# ----------------------------------------
# NEW ↑
# ----------------------------------------

def tensor2pil(t_image: torch.Tensor)  -> Image:
    if t_image.dtype != torch.float32:
        t_image = t_image.float()
    return Image.fromarray(
        np.clip(
            255.0 * t_image.cpu().numpy().squeeze(),
            0,
            255
        ).astype(np.uint8)
    )

def pil2tensor(image:Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def is_valid_mask(tensor:torch.Tensor) -> bool:
    return not bool(torch.all(tensor == 0).item())

def image2mask(image:Image) -> torch.Tensor:
    if image.mode == "L":
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])
    else:
        image = image.convert("RGB").split()[0]
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])

def tensor2np(tensor: torch.Tensor) -> List[np.ndarray]:
    if len(tensor.shape) == 3:  # Single image
        return np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    else:  # Batch of images
        return [np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8) for t in tensor]

def mask2image(mask:torch.Tensor)  -> Image:
    masks = tensor2np(mask)
    for m in masks:
        _mask = Image.fromarray(m).convert("L")
        _image = Image.new("RGBA", _mask.size, color="white")
        _image = Image.composite(
            _image, Image.new("RGBA", _mask.size, color="black"), _mask)
    return _image

def fit_resize_image(image:Image, target_width:int, target_height:int, fit:str, resize_sampler:str, background_color:str = "#000000") -> Image:
    image = image.convert("RGB")
    orig_width, orig_height = image.size
    if image is not None:
        if fit == "letterbox":
            if orig_width / orig_height > target_width / target_height:  # Wider, with black bars at the top and bottom.
                fit_width = target_width
                fit_height = int(target_width / orig_width * orig_height)
            else:  # Slimmer, with black bars on the left and right.
                fit_height = target_height
                fit_width = int(target_height / orig_height * orig_width)
            fit_image = image.resize((fit_width, fit_height), resize_sampler)
            ret_image = Image.new("RGB", size=(target_width, target_height), color=background_color)
            ret_image.paste(fit_image, box=((target_width - fit_width)//2, (target_height - fit_height)//2))
        elif fit == "crop":
            if orig_width / orig_height > target_width / target_height:  # Wider — Crop Left and Right
                fit_width = int(orig_height * target_width / target_height)
                fit_image = image.crop(
                    ((orig_width - fit_width)//2, 0, (orig_width - fit_width)//2 + fit_width, orig_height))
            else:   # Slimmer—trimmed at the top and bottom.
                fit_height = int(orig_width * target_height / target_width)
                fit_image = image.crop(
                    (0, (orig_height-fit_height)//2, orig_width, (orig_height-fit_height)//2 + fit_height))
            ret_image = fit_image.resize((target_width, target_height), resize_sampler)
        else:
            ret_image = image.resize((target_width, target_height), resize_sampler)
    return  ret_image

def generate_text_image(width:int, height:int, text:str, font_file:str, text_scale:float=1, font_color:str="#FFFFFF",) -> Image:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font_size = int(width / len(text) * text_scale)
    font = ImageFont.truetype(font_file, font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = int((width - text_width) / 2)
    y = int((height - text_height) / 2) - int(font_size / 2)
    draw.text((x, y), text, font=font, fill=font_color)
    return image


def avoid_naming_collisions(folder: str, basename: str, ext: str) -> str:
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
            ts = datetime.datetime.now().strftime("%m%d%H%M%S%f")
            return f"{basename}_{ts}{ext}"


def get_output_path(node_name: str, filename_prefix: str, output_path: str) -> tuple[str, str, str]:
    """Resolve output folder and compute filename with variable substitution."""

    # Variable substitution
    if "%HMSf%" in filename_prefix:
        filename_prefix = filename_prefix.replace(
            "%HMSf%",
            f"{datetime.datetime.now():%m%d%H%M%S%f}"
        )

    # Split into subfolder and filename
    subfolder = os.path.dirname(os.path.normpath(filename_prefix))
    filename_base = os.path.basename(os.path.normpath(filename_prefix))

    full_output_folder = os.path.join(output_path, subfolder)

    # Security check: the target ideally should live within the outputs directory.
    # An equality test would flag every legitimate subfolder, so this
    # checks containment instead and rejects ".." traversal.
    def _security_check(base: str, target: str) -> bool:
        base = os.path.normcase(os.path.abspath(base))
        target = os.path.normcase(os.path.abspath(target))
        try:
            return os.path.commonpath([base, target]) == base
        except ValueError:
            # Different drives on Windows
            return False

    if not _security_check(output_path, full_output_folder):
        log(
            f"{node_name}💾 is saving outside the output directory -> {full_output_folder}",
            message_type="warning",
        )

    # Ensure folder exists
    os.makedirs(full_output_folder, exist_ok=True)

    return full_output_folder, filename_base, subfolder

