"""
Image Scaler

┌───────────────────────────────────────────────────┐
│                   Image Scaler                    │
├───────────────────────────────────────────────────┤
│ ○ Image                              Image     ○  │
│ ○ Mask                                Mask     ○  │
│ ○ VAE                               Latent     ○  │
│                                      Width     ○  │
│                                     Height     ○  │
│ <→ INPUT>    Grow Mask By      Default = 6        │
│ <▼ DROPDOWN> Fit                                  │
│ <▼ DROPDOWN> Scaling Method                       │
│ <▼ DROPDOWN> Base Model (latent)                  │
│ <→ INPUT>    Background Color  Default = auto     │
│ ─────────────── SWITCHES (pick one) ───────────── │
│ <↔ BOOLEAN>  Enable: Round To Multiple            │
│   <↔ BOOLEAN>  Rounding    (Up / Down)            │
│   <▼ DROPDOWN> Round To Multiple                  │
│ <↔ BOOLEAN>  Enable: Scale To Total Pixels        │
│   <→ INPUT>    Megapixels        0.01 - 16.0      │
│   <→ INPUT>    Resolution Steps  1 - 256          │
│ <↔ BOOLEAN>  Enable: Manual Size                  │
│   <→ INPUT>    Width   512 - 16384                │
│   <→ INPUT>    Height  512 - 16384                │
└───────────────────────────────────────────────────┘

Only one of the three switches may be active at a time. The companion
web/js/fg_image_scale.js enforces this in the UI (turning one on turns the
others off, and greys/hides the options that don't apply). The Python side
enforces it again for API calls, where the JS never runs.

With no switch enabled the image passes through untouched — a latent is
still produced, which is the usual reason to keep this node in a graph.

Background Color:
    "auto" (or "-1", or blank) samples the image's own border pixels and uses
    the dominant edge colour. Anything PIL understands also works: "#RRGGBB",
    "rgb(12,34,56)", "white". Only used by the "letterbox" fit.

Image:
    Batches are supported. All frames are resized to the same target, which is
    derived from the FIRST image, so mixed-size batches will be forced to the
    first frame's aspect handling.

Optional: Mask, VAE
Output: Image, Mask, Latent, Width, Height
"""

import math
import torch
import numpy as np
from PIL import Image
import comfy.model_management
from ._fg_helperfunctions import (
    log,
    tensor2pil,
    pil2tensor,
    image2mask,
    unpack_images,
    unpack_masks,
    fit_resize_image,
    generate_latent_image_data,
    MODEL_TYPES,
    SCALING_METHODS
)

FIT_MODES     = ["crop", "fill", "letterbox"]
MULTIPLE_LIST = ["8", "16", "32", "64", "128", "256", "512", "None"]
MEGAPIXEL     = 1024 * 1024                       # matches ComfyUI core's ImageScaleToTotalPixels
# Letterbox padding colour. None = sample the image's own border (recommended).
# Set it to any colour PIL understands ("#FFFFFF", "white", "rgb(0,0,0)") to force one.
LETTERBOX_COLOR = None


# --------------------------------------------------------------------------- #
# Background colour helpers
# --------------------------------------------------------------------------- #
def sample_background_color(pil_image: Image.Image, border_frac: float = 0.03, min_border: int = 2) -> str:
    """
    Guess an image's background colour from its border pixels.

    Takes a frame of pixels around the edge, buckets them into 32 levels per
    channel, finds the most populated bucket and averages the real pixels that
    landed in it. Bucketing first means a slightly noisy or gradient border
    still resolves to one colour instead of whatever single value happens to be
    the median.
    """
    im = pil_image.convert("RGB")
    width, height = im.size
    if width < 2 or height < 2:
        return "#FFFFFF"

    border = max(min_border, int(round(min(width, height) * border_frac)))
    border = max(1, min(border, min(width, height) // 2))

    arr = np.asarray(im, dtype=np.uint8)
    edges = np.concatenate(
        [
            arr[:border, :, :].reshape(-1, 3),
            arr[-border:, :, :].reshape(-1, 3),
            arr[:, :border, :].reshape(-1, 3),
            arr[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    if edges.size == 0:
        return "#FFFFFF"

    bucket = (edges >> 3).astype(np.int32)                      # 0-31 per channel
    keys = bucket[:, 0] * 1024 + bucket[:, 1] * 32 + bucket[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    winner = values[int(np.argmax(counts))]
    mean = edges[keys == winner].mean(axis=0)
    red, green, blue = (int(round(float(c))) for c in mean)
    return f"#{red:02X}{green:02X}{blue:02X}"


# --------------------------------------------------------------------------- #
# Size maths
# --------------------------------------------------------------------------- #
def round_dimension(value: float, multiple: int, up: bool = True) -> int:
    """Round a single dimension to a multiple, never returning less than one multiple."""
    if multiple is None or multiple <= 1:
        return int(max(1, round(value)))
    if up:
        result = math.ceil(value / multiple) * multiple
    else:
        result = (int(value) // multiple) * multiple
    return int(max(multiple, result))


def snap_to_step(value: float, step: int) -> int:
    """Round a dimension to the NEAREST multiple of step."""
    if step is None or step <= 1:
        return int(max(1, round(value)))
    return int(max(step, round(value / step) * step))


def size_for_megapixels(width: int, height: int, megapixels: float, steps: int) -> tuple:
    """Keep the aspect ratio, hit a total pixel budget, then snap to the step size."""
    pixels = max(1, int(width) * int(height))
    target = max(1.0, float(megapixels) * MEGAPIXEL)
    scale = math.sqrt(target / float(pixels))
    return snap_to_step(width * scale, steps), snap_to_step(height * scale, steps)


class FG_ImageScaler:

    def __init__(self):
        self.NODE_NAME = "Image Scale"
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
                "vae" : ("VAE", {"tooltip": "🔣 VAE: Used to encode the reference images into latent the model understands."}),

                # ---------------- options ----------------
                "grow_mask_by"    : ("INT", {
                    "default": 6, "min": 0, "max": 64, "step": 1,
                    "tooltip": "🎭 Grows the mask a few pixels before encoding so inpaint seams stay clean. Needs a mask AND a VAE."}),
                "fit"             : (FIT_MODES, {
                    "default": "fill",
                    "tooltip": "🖼️ crop = fill the frame and trim the overflow. fill = stretch to fit. letterbox = fit inside and pad with the image's own border colour."}),
                "scaling_method"  : (list(SCALING_METHODS.keys()), {
                    "default": "lanczos",
                    "tooltip": "🪄 Resampling filter. lanczos for downscales, bicubic for upscales, nearest for pixel art."}),
                "base_model"      : (list(MODEL_TYPES.keys()), {
                    "default": "SDXL",
                    "tooltip": "🌀 Latent shape to build when no VAE is connected. SDXL & FLUX differ."}),

                # ---------------- switches ----------------
                "enable_round_to_multiple": ("BOOLEAN", {
                    "default": False, "label_on": "Round To Multiple: ON", "label_off": "Round To Multiple: off",
                    "tooltip": "🔘 Keep the original size, nudged to the nearest multiple. Turning this on turns the other two switches off."}),
                "rounding"          : ("BOOLEAN", {
                    "default": True, "label_on": "Round Up", "label_off": "Round Down"}),
                "round_to_multiple" : (MULTIPLE_LIST, {"default": "64"}),

                "enable_scale_to_megapixels": ("BOOLEAN", {
                    "default": False, "label_on": "Total Pixels: ON", "label_off": "Total Pixels: off",
                    "tooltip": "🔘 Scale to a pixel budget while keeping the aspect ratio. Turning this on turns the other two switches off."}),
                "megapixels"       : ("FLOAT", {
                    "default": 1.00, "min": 0.01, "max": 16.00, "step": 0.01,
                    "tooltip": "📐 Target megapixels (1 MP = 1024×1024)."}),
                "resolution_steps" : ("INT", {
                    "default": 64, "min": 1, "max": 256, "step": 1,
                    "tooltip": "📏 Snap the result to multiples of this. 1 = no snapping."}),

                "enable_manual_size": ("BOOLEAN", {
                    "default": False, "label_on": "Manual Size: ON", "label_off": "Manual Size: off",
                    "tooltip": "🔘 Scale to exact numbers. Turning this on turns the other two switches off."}),
                "desired_width"    : ("INT", {"default": 1024, "min": 512, "max": 16384, "step": 8}),
                "desired_height"   : ("INT", {"default": 1024, "min": 512, "max": 16384, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "LATENT", "INT", "INT",)
    RETURN_NAMES = ("Image", "Mask", "Latent", "Width", "Height",)
    FUNCTION = "scale_image"
    CATEGORY = "Farrenzo's Garbage/Image/Utils"
    DESCRIPTION = "Resize an image (and its mask) by rounding, by total pixels, or to exact dimensions, and hand back a matching latent."

    # ----------------------------------------------------------------- #
    def _pick_mode(self, round_on: bool, megapixels_on: bool, manual_on: bool):
        """
        Exactly one switch wins. The JS keeps the UI honest; this keeps API
        calls and hand-edited workflows honest.
        """
        active = [
            # Order matters: it's the tie-break when a hand-edited or
            # API-authored graph turns on more than one. Same order as the
            # widgets, so the JS resolves ties the same way.
            name for name, flag in (
                ("round", round_on),
                ("megapixels", megapixels_on),
                ("manual", manual_on),
            ) if flag
        ]
        if not active:
            return None
        if len(active) > 1:
            log(
                f"{self.NODE_NAME}: {len(active)} scaling switches were on ({', '.join(active)}). "
                f"Using '{active[0]}' and ignoring the rest.",
                message_type="warning",
            )
        return active[0]

    def _target_size(self, mode, og_width, og_height, rounding, round_to_multiple,
                     megapixels, resolution_steps, desired_width, desired_height):
        """Return (width, height, should_scale, message)."""
        if mode is None:
            return og_width, og_height, False, "No scaling switch enabled, passing the image through."

        if mode == "manual":
            width, height = max(8, int(desired_width)), max(8, int(desired_height))
            return width, height, True, f"Scaling to {width}x{height}."

        if mode == "megapixels":
            width, height = size_for_megapixels(og_width, og_height, megapixels, int(resolution_steps))
            return (
                width, height, True,
                f"Scaling to ~{float(megapixels):.2f} MP in steps of {int(resolution_steps)} -> {width}x{height}.",
            )

        # mode == "round"
        multiple = None if str(round_to_multiple) == "None" else int(round_to_multiple)
        if multiple is None:
            return og_width, og_height, False, "Round to multiple is set to 'None', passing the image through."
        up = bool(rounding)
        width  = round_dimension(og_width, multiple, up)
        height = round_dimension(og_height, multiple, up)
        return (
            width, height, True,
            f"Rounding {'up' if up else 'down'} to the nearest multiple of {multiple} -> {width}x{height}.",
        )

    # ----------------------------------------------------------------- #
    def scale_image(
        self,
        image,
        fit                        = "fill",
        scaling_method             = "lanczos",
        base_model                 = "SDXL",
        vae                        = None,
        mask                       = None,
        grow_mask_by               = 6,
        enable_round_to_multiple   = True,
        rounding                   = True,
        round_to_multiple          = "64",
        enable_scale_to_megapixels = False,
        megapixels                 = 1.00,
        resolution_steps           = 64,
        enable_manual_size         = False,
        desired_width              = 1024,
        desired_height             = 1024,
    ):
        resize_sampler = SCALING_METHODS[scaling_method]
        output_images, output_masks = [], []
        latent, latent_info = None, "empty"

        # ---------------- unpack ----------------
        unpacked_images, og_width, og_height = unpack_images(images=image)

        unpacked_masks = []
        if mask is not None:
            unpacked_masks, mask_width, mask_height = unpack_masks(masks=mask)
            if mask_width == 0 or mask_height == 0:
                log(f"{self.NODE_NAME}: Input mask is empty, ignoring it.", message_type="warning")
                unpacked_masks = []
            elif (og_width != mask_width) or (og_height != mask_height):
                log(
                    f"{self.NODE_NAME}: First mask ({mask_width}x{mask_height}) doesn't match the first image "
                    f"({og_width}x{og_height}). Dropping the mask.",
                    message_type="warning",
                )
                unpacked_masks = []

        # ---------------- decide the target ----------------
        mode = self._pick_mode(
            bool(enable_round_to_multiple),
            bool(enable_scale_to_megapixels),
            bool(enable_manual_size),
        )
        target_width, target_height, scale, message = self._target_size(
            mode, og_width, og_height, rounding, round_to_multiple,
            megapixels, resolution_steps, desired_width, desired_height,
        )
        log(f"{self.NODE_NAME}: {message}")

        # ---------------- resize ----------------
        if not scale:
            output_images = unpacked_images
            output_masks = list(unpacked_masks)
        else:
            for frame in unpacked_images:
                pil_frame = tensor2pil(frame).convert("RGB")
                bg = "#000000"
                if fit == "letterbox":
                    bg = LETTERBOX_COLOR or sample_background_color(pil_frame)
                pil_frame = fit_resize_image(pil_frame, target_width, target_height, fit, resize_sampler, bg)
                output_images.append(pil2tensor(pil_frame))

            for frame_mask in unpacked_masks:
                pil_mask = tensor2pil(frame_mask).convert("L")
                pil_mask = fit_resize_image(pil_mask, target_width, target_height, fit, resize_sampler).convert("L")
                output_masks.append(image2mask(pil_mask))

        image_batch = torch.cat(output_images, dim=0)
        mask_batch = torch.cat(output_masks, dim=0) if len(output_masks) > 0 else None

        # ---------------- latent ----------------
        if vae is None:
            latent_info, latent = generate_latent_image_data(
                width      = target_width,
                height     = target_height,
                batch_size = image_batch.shape[0],
                model_type = base_model,
            )
            log(f"{self.NODE_NAME}: No VAE connected. Generated an {latent_info} latent of {target_width}x{target_height}.")
        elif mask_batch is None:
            latent_info, latent = generate_latent_image_data(
                width  = target_width,
                height = target_height,
                vae    = vae,
                image  = image_batch,
            )
            log(f"{self.NODE_NAME}: Found a VAE but no mask, encoding the image batch into a latent.")
        else:
            # vae_encode_inpainter wants image and mask batches of the same length.
            if mask_batch.shape[0] != image_batch.shape[0]:
                log(
                    f"{self.NODE_NAME}: {image_batch.shape[0]} images vs {mask_batch.shape[0]} masks. "
                    f"Encoding the first of each.",
                    message_type="warning",
                )
                encode_image, encode_mask = image_batch[:1], mask_batch[:1]
            else:
                encode_image, encode_mask = image_batch, mask_batch
            latent_info, latent = generate_latent_image_data(
                width           = target_width,
                height          = target_height,
                batch_size      = encode_image.shape[0],
                model_type      = base_model,
                vae             = vae,
                image           = encode_image,
                mask            = encode_mask,
                mask_growth_val = grow_mask_by,
            )
            log(f"{self.NODE_NAME}: Found a mask and a VAE, encoding a latent for inpainting.")

        log(
            f"{self.NODE_NAME} Processed {image_batch.shape[0]} image(s), "
            f"{0 if mask_batch is None else mask_batch.shape[0]} mask(s) & an {latent_info} latent "
            f"at {target_width}x{target_height}.",
            message_type="finish",
        )
        return (image_batch, mask_batch, latent, target_width, target_height)
