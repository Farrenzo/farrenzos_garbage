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
import comfy.model_management as comfy_mm

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
    device = comfy_mm.intermediate_device()
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
                dtype=comfy_mm.intermediate_dtype()
            )
        }
        latent_info = "empty"
    elif vae is not None and mask is None:
        # encode_auto instead of vae.encode: it measures free VRAM and only
        # tiles when a single pass won't fit. Every node that builds a latent
        # comes through here, so they all get it for free.
        samples, tile_px = encode_auto(vae, image)
        latent = {"samples": samples}
        latent_info = f"image (tiled at {tile_px}px)" if tile_px else "image"
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
    # The masked pixels are just pixels by this point, so the same auto-tiling
    # applies. This was the one encode in the pack still going through in a
    # single pass regardless of size. Return shape is unchanged for callers;
    # the tile decision is only worth a log line.
    t, tile_px = encode_auto(vae, pixels)
    if tile_px:
        log(f"Inpaint encode tiled at {tile_px}px ({pixels.shape[2]}x{pixels.shape[1]}).")

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


def clear_memory(purge_cache: bool = False, purge_models: bool = False, keep: float = 0.2, nuclear: bool = False):
    """
    keep:    fraction of total VRAM to leave loaded (0.2 == the old 0.8 free target).
             Ignored when nuclear=True.
    nuclear: request an absurdly large free_memory target (1e30) instead of a
             fraction of total VRAM. ComfyUI's free_memory(small_or_zero, device)
             is effectively a no-op for reclaim on some backends/versions — it
             marks weights as evictable but leaves their .data resident. Asking
             for far more than could ever be needed forces the memory manager to
             evict everything it possibly can rather than leaving weights
             "soft-unloaded" but still occupying VRAM. Use this for an explicit
             "empty the card" call; use the plain keep-based version for routine
             headroom management where leaving some models warm is fine.

    Order matters: models are unloaded BEFORE the cache/backend purge runs, so
    their tensors are already dereferenced by the time empty_cache/free_memory
    are asked to reclaim anything. Purging cache first (the old order) could
    run while model params were still registered as loaded.
    """
    if purge_models:
        comfy_mm.unload_all_models()

    if purge_cache:
        gc.collect()
        for name, mod in get_accelerators():
            for device in get_devices(name, mod):
                try:
                    target = 1e30 if nuclear else (
                        comfy_mm.get_total_memory(device) * (1.0 - keep)
                    )
                    comfy_mm.free_memory(target, device)
                except Exception as e:
                    log(f"⚠️ free_memory failed on {device}: {e}", message_type="warning")
                purge_backend(name, mod, device)
        gc.collect()

    log(f"👝 Memory purged{' (nuclear)' if nuclear else ''}.", message_type="finish")

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

# --------------------------------------------------------------------------- #
# Shared VAE tiling decisions, for encode AND decode
# --------------------------------------------------------------------------- #
# This logic started life inside fg_load_vae.py's decode path. It lives here so
# the encode side can use it too -- notably generate_latent_image_data below,
# and fg_image_scale, which can hand the VAE a model-upscaled 4096x4096 image.
#
# The important idea, and the reason this is worth sharing rather than
# reimplementing: AUTO's first job is deciding whether to tile AT ALL, not
# picking a size. For 2D VAEs comfy's decode_tiled_ / encode_tiled_ run
# tiled_scale three times (tile//2 x tile*2, tile*2 x tile//2, tile x tile) and
# average the results to hide seams, so tiling an image VAE costs roughly 3x no
# matter how the tiles are sized. Skipping it is worth far more than any amount
# of tile tuning. decode_tiled_3d runs once, so video VAEs pay no such penalty.
#
# Units, which are easy to get wrong (see comfy/sd.py):
#   * decode_tiled takes tile sizes in LATENT units.
#   * encode_tiled takes them in PIXELS.
# Stock VAEDecodeTiled divides by spacial_compression_decode(); stock
# VAEEncodeTiled does not. Everything here is in pixels and converts on the way
# in, so callers never have to think about it.
#
#   tile_size / overlap / temporal_size / temporal_overlap:
#      -1  = auto (default)
#       0  = force off
#      >0  = explicit, in pixels (frames for the temporal pair)

AUTO = -1
# Fraction of free VRAM an untiled pass may claim before we tile instead.
HEADROOM = 0.75
TILE_LADDER = (2048, 1536, 1024, 768, 512, 384, 256, 192, 128)


def bounded_shape(shape, tile_x, tile_y, tile_t):
    """Shape of a single tile, for feeding comfy's memory estimators."""
    s = list(shape)
    if len(s) == 5:      # B C T H W
        if tile_t:
            s[2] = min(s[2], tile_t)
        if tile_y:
            s[3] = min(s[3], tile_y)
        if tile_x:
            s[4] = min(s[4], tile_x)
    elif len(s) == 4:    # B C H W
        if tile_y:
            s[2] = min(s[2], tile_y)
        if tile_x:
            s[3] = min(s[3], tile_x)
    return tuple(s)


def auto_tile(vae, shape, estimator, unit_divisor, tile_t=None):
    """Return a tile size in PIXELS, or 0 for 'do not tile'.

    shape is in the estimator's own units (latent for decode, pixel for
    encode). unit_divisor converts pixels into those units.
    """
    try:
        free = comfy_mm.get_free_memory(vae.device)
        full = estimator(shape, vae.vae_dtype)
    except Exception:
        return 0

    budget = free * HEADROOM
    if full <= budget:
        return 0  # fits whole -- always cheaper than any tiling

    longest = max(shape[-2], shape[-1]) * unit_divisor

    for px in TILE_LADDER:
        if px > longest:
            continue
        units = max(1, px // unit_divisor)
        try:
            cost = estimator(bounded_shape(shape, units, units, tile_t), vae.vae_dtype)
        except Exception:
            return 512
        if cost <= budget:
            return px

    return TILE_LADDER[-1]


def resolve(value, auto_value):
    return auto_value if value == AUTO else value


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def encode_auto(vae, pixels, tile_size=AUTO, overlap=AUTO,
                temporal_size=AUTO, temporal_overlap=AUTO):
    """Encode pixels (B, H, W, C) to a latent, tiling only if it won't fit.

    Returns (latent_tensor, tile_px) where tile_px is 0 when no tiling was
    used -- handy for logging what actually happened.
    """
    pixels = pixels[..., :3]
    shape = (pixels.shape[0], 3, pixels.shape[1], pixels.shape[2])
    size = resolve(tile_size, auto_tile(vae, shape, vae.memory_used_encode, 1))

    if size <= 0:
        return vae.encode(pixels), 0

    ov = min(resolve(overlap, max(32, size // 8)), size // 4)
    t_size = resolve(temporal_size, 64)
    t_overlap = resolve(temporal_overlap, 8)

    # encode_tiled already works in pixels -- no conversion.
    return vae.encode_tiled(pixels, tile_x=size, tile_y=size, overlap=ov,
                            tile_t=t_size, overlap_t=t_overlap), size


def decode_auto(vae, latent, tile_size=AUTO, overlap=AUTO,
                temporal_size=AUTO, temporal_overlap=AUTO):
    """Decode a latent to pixels, tiling only if it won't fit.

    Returns (image_tensor, tile_px), tile_px 0 when untiled.
    """
    compression = vae.spacial_compression_decode() or 8
    frames = latent.shape[2] if latent.ndim == 5 else 1

    t_size = resolve(temporal_size, 0 if frames <= 1 else 64)
    t_overlap = resolve(temporal_overlap, 8)

    size = resolve(
        tile_size,
        auto_tile(vae, latent.shape, vae.memory_used_decode, compression,
                  tile_t=(t_size or None)),
    )

    if size <= 0:
        return vae.decode(latent), 0

    ov = min(resolve(overlap, max(32, size // 8)), size // 4)

    # decode_tiled wants latent units; everything here is in pixels.
    tile_lat = max(1, size // compression)
    ov_lat = max(0, ov // compression)

    t_comp = vae.temporal_compression_decode()
    if t_comp is not None and t_size:
        tt = max(2, t_size // t_comp)
        to = max(1, min(tt // 2, t_overlap // t_comp))
    else:
        tt = to = None

    return vae.decode_tiled(latent, tile_x=tile_lat, tile_y=tile_lat,
                            overlap=ov_lat, tile_t=tt, overlap_t=to), size
