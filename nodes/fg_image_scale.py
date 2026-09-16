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
│   <↔ BOOLEAN>  Use Upscale Model                  │
│     <→ INPUT>    Max Longest Length   default 4096│
│     <▼ DROPDOWN> Model Multiple       default 64  │
│     <▼ DROPDOWN> Upscale Model                    │
└───────────────────────────────────────────────────┘

Only one of the three switches may be active at a time. The companion
web/js/fg_image_scale.js enforces this in the UI (turning one on turns the
others off, and greys/hides the options that don't apply). The Python side
enforces it again for API calls, where the JS never runs.

All three switches default to OFF. With none enabled the image passes through
untouched — a latent is still produced, which is the usual reason to keep this
node in a graph.

Use Upscale Model (manual size only):
    Absorbs the old FG_ModelImageScaler node. Detail survives a downscale far
    better than it survives an upscale, so instead of interpolating straight to
    the target this runs a real upscale model first, overshooting deliberately,
    then resamples down.

    The intermediate size is the largest WHOLE multiple of the source whose
    longest side still fits under max_longest_length, snapped to model_multiple:

        512x512,  cap 4096 -> k=8 -> 4096x4096  (8x512 = 4096, fits)
        768x1344, cap 4096 -> k=3 -> 2304x4032  (4x1344 = 5376, too big)

    The model runs as many passes as it takes to reach that size (a 4x model on
    768x1344 lands at 3072x5376 in one), the result is resampled down to the
    intermediate, and then down again to the width/height you asked for. When
    your target IS the intermediate, that last step is skipped rather than
    resampling to the same numbers twice.

    Whole multiples are what your examples describe, and they are also the only
    ratios at which an upscale model's output grid lines up with the source
    pixel grid, so nothing gets resampled twice on the way up.

VAE encoding:
    Handled by _fg_vae_tiling.encode_auto, shared with the decode side. It
    measures free VRAM and only tiles when a single pass won't fit, which
    matters a great deal here: the model path hands the VAE images several
    times larger than anything the other modes produce.

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
import folder_paths
import comfy.utils as c_utils
import comfy.model_management
from ._fg_helperfunctions import (
    log,
    tensor2pil,
    pil2tensor,
    image2mask,
    encode_auto,
    unpack_masks,
    unpack_images,
    fit_resize_image,
    generate_latent_image_data,

    MODEL_TYPES,
    SCALING_METHODS
)


# spandrel ships with ComfyUI, but the node should still load without it --
# only the use_model path needs it, and an ImportError at module scope would
# take the whole node pack down with it.
try:
    from spandrel import ModelLoader, ImageModelDescriptor
    SPANDREL_AVAILABLE = True
except Exception:
    ModelLoader = ImageModelDescriptor = None
    SPANDREL_AVAILABLE = False

FIT_MODES     = ["crop", "fill", "letterbox"]
MULTIPLE_LIST = ["8", "16", "32", "64", "128", "256", "512", "None"]
# Upscale models are heavy and usually reused across runs, so cache the loaded
# one at module scope -- ComfyUI may build a fresh node instance per execution.
_MODEL_CACHE = {}
MEGAPIXEL     = 1024 * 1024  # matches ComfyUI core's ImageScaleToTotalPixels
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


def model_intermediate_size(src_width: int, src_height: int,
                            max_longest: int, multiple) -> tuple:
    """How big to blow the image up before coming back down.

    The largest WHOLE multiple of the source whose longest side still fits
    inside max_longest, then floored to `multiple`. Whole multiples keep the
    upscale model's output grid aligned with the source pixel grid, and they
    are what the worked examples describe:

        512x512,  4096 -> 8x -> 4096x4096
        768x1344, 4096 -> 3x -> 2304x4032

    Only the LONGEST side is bound by the cap, so only it gets floored to the
    multiple -- rounding it up could push past max_longest. The short side is
    rounded to the NEAREST multiple instead, because flooring both independently
    quietly skews the aspect: 1920x1080 at 2x is 3840x2160, and flooring 2160 to
    a multiple of 64 gives 2112, turning 1.78 into 1.82. Nobody asked for that,
    and it would come back as a stretch or a crop at the final resize.
    """
    src_width, src_height = int(src_width), int(src_height)
    longest = max(src_width, src_height)
    if longest <= 0:
        return src_width, src_height

    factor = max(1, int(max_longest) // longest)
    if factor == 1:
        # The source already fills (or overflows) the cap, so there is no
        # headroom to blow up into. Hand the source straight back, unsnapped --
        # snapping here could only return something SMALLER than the source,
        # and on an oversized source it would return something over the cap.
        # The caller reads mid <= source as "skip the model".
        return src_width, src_height

    width, height = src_width * factor, src_height * factor

    if multiple:
        multiple = int(multiple)
        floor_to = lambda v: max(multiple, (v // multiple) * multiple)
        near_to = lambda v: max(multiple, int(round(v / multiple)) * multiple)
        if width >= height:
            width, height = floor_to(width), near_to(height)
        else:
            width, height = near_to(width), floor_to(height)

    return width, height


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

                # ------- sub-switch of manual size: upscale with a model -------
                "use_model"         : ("BOOLEAN", {
                    "default": False, "label_on": "Use Upscale Model: ON", "label_off": "Use Upscale Model: off",
                    "tooltip": "🔍 Blow the image up with an upscale model first, then resample down to your "
                               "width/height. Detail survives a downscale much better than an upscale. "
                               "Off = go straight there with the scaling method above."}),
                "max_longest_length": ("INT", {
                    "default": 4096, "min": 512, "max": 16384, "step": 64,
                    "tooltip": "📏 Ceiling for the blown-up intermediate's longest side. The source is "
                               "multiplied by the largest whole number that still fits under this "
                               "(512 with a 4096 cap -> 8x -> 4096; 1344 -> 3x -> 4032)."}),
                "model_multiple"    : (MULTIPLE_LIST, {
                    "default": "64",
                    "tooltip": "📐 Snap the intermediate size down to a multiple of this. 'None' = no snapping."}),
                "upscale_model"     : (folder_paths.get_filename_list("upscale_models") or ["None"], {
                    "tooltip": "🧠 The model that does the blowing up. Only used when Use Upscale Model is on."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "LATENT", "INT", "INT",)
    RETURN_NAMES = ("Image", "Mask", "Latent", "Width", "Height",)
    FUNCTION = "scale_image"
    CATEGORY = "Farrenzo's Garbage/Image/Utils"
    DESCRIPTION = ("Resize an image (and its mask) by rounding, by total pixels, or to exact "
                   "dimensions — optionally via an upscale model — and hand back a matching latent.")
    SEARCH_ALIASES = ["scale", "resize", "upscale", "upscale model", "image scale", "model upscale"]

    # ----------------------------------------------------------------- #
    # Upscale-model path (absorbed from the old FG_ModelImageScaler)
    # ----------------------------------------------------------------- #
    def _load_model(self, model_name):
        if not SPANDREL_AVAILABLE:
            raise RuntimeError(
                f"{self.NODE_NAME}: spandrel isn't importable, so upscale models can't be "
                f"loaded. Turn Use Upscale Model off, or repair the ComfyUI install."
            )
        model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
        cached = _MODEL_CACHE.get(model_path)
        if cached is not None:
            return cached

        state_dict = c_utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in state_dict:
            state_dict = c_utils.state_dict_prefix_replace(state_dict, {"module.": ""})
        model = ModelLoader().load_from_state_dict(state_dict).eval()

        if not isinstance(model, ImageModelDescriptor):
            raise Exception(f"{self.NODE_NAME}: '{model_name}' is not a single-image upscale model.")

        # One model at a time: these run 100s of MB and holding several would
        # defeat the point of freeing memory before each pass.
        _MODEL_CACHE.clear()
        _MODEL_CACHE[model_path] = model
        return model

    def _upscale_w_model(self, model, pic):
        """One pass of the model, tiled, halving the tile on OOM."""
        device = comfy.model_management.get_torch_device()
        memory_required = comfy.model_management.module_size(model.model)
        memory_required += (512 * 512 * 3) * pic.element_size() * max(model.scale, 1.0) * 384.0
        # The 384.0 is an estimate of how much some of these models take,
        # TODO: make it more accurate
        memory_required += pic.nelement() * pic.element_size()
        comfy.model_management.free_memory(memory_required, device)
        model.to(device)
        in_img = pic.movedim(-1, -3).to(device)

        tile = 512
        overlap = 32

        oom = True
        try:
            while oom:
                try:
                    steps = in_img.shape[0] * c_utils.get_tiled_scale_steps(
                        in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                    pbar = c_utils.ProgressBar(steps)
                    s = c_utils.tiled_scale(
                        in_img, lambda a: model(a), tile_x=tile, tile_y=tile,
                        overlap=overlap, upscale_amount=model.scale, pbar=pbar)
                    oom = False
                except Exception as e:
                    comfy.model_management.raise_non_oom(e)
                    tile //= 2
                    if tile < 128:
                        raise e
        finally:
            model.to("cpu")
        return torch.clamp(s.movedim(-3, -1), min=0, max=1.0)

    def _model_upscale_to(self, model, image_batch, goal_width, goal_height, max_longest):
        """Run model passes until the batch reaches goal size, then clamp to it.

        A model has a fixed scale (usually 2x or 4x), so landing exactly on the
        goal is luck. Overshooting and coming down is the whole point -- that's
        where the detail comes from -- so passes run until the goal is met or
        beaten, then a single resample brings it to the goal.
        """
        scale = float(getattr(model, "scale", 1.0) or 1.0)
        current = image_batch
        passes = 0

        if scale <= 1.0:
            log(f"{self.NODE_NAME}: The upscale model reports a scale of {scale}x, running one pass anyway.",
                message_type="warning")
            current = self._upscale_w_model(model, current)
            passes = 1
        else:
            # Never let a pass land more than 2x past the cap on each side.
            # Reaching a 4096 goal from 512 with a 4x model means passing
            # through 8192, which is fine; 16384 is not.
            pixel_budget = (int(max_longest) * 2) ** 2
            max_passes = 4
            while passes < max_passes:
                height, width = current.shape[1], current.shape[2]
                if width >= goal_width and height >= goal_height:
                    break
                next_pixels = (width * scale) * (height * scale)
                if passes > 0 and next_pixels > pixel_budget:
                    log(
                        f"{self.NODE_NAME}: Stopping at {width}x{height} — another {scale:g}x pass would "
                        f"need {next_pixels / 1e6:.0f} MP, past the {pixel_budget / 1e6:.0f} MP ceiling. "
                        f"Resampling up the rest of the way.",
                        message_type="warning",
                    )
                    break
                current = self._upscale_w_model(model, current)
                passes += 1

        height, width = current.shape[1], current.shape[2]
        log(f"{self.NODE_NAME}: {passes} model pass(es) -> {width}x{height}.")

        if (width, height) != (goal_width, goal_height):
            samples = current.movedim(-1, 1)
            samples = c_utils.common_upscale(samples, goal_width, goal_height, "lanczos", "disabled")
            current = samples.movedim(1, -1)
            log(f"{self.NODE_NAME}: Resampled the model output to the intermediate {goal_width}x{goal_height}.")

        return current

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
        # Was True here while INPUT_TYPES said False -- an API call that left
        # it out got rounding it never asked for. All three default off now.
        enable_round_to_multiple   = False,
        rounding                   = True,
        round_to_multiple          = "64",
        enable_scale_to_megapixels = False,
        megapixels                 = 1.00,
        resolution_steps           = 64,
        enable_manual_size         = False,
        desired_width              = 1024,
        desired_height             = 1024,
        use_model                  = False,
        max_longest_length         = 4096,
        model_multiple             = "64",
        upscale_model              = None,
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

        # ---------------- optional model upscale ----------------
        # Only under manual size: the other modes derive their target from the
        # source, so blowing the source up first would change the answer.
        model_used = False
        if mode == "manual" and bool(use_model):
            if not upscale_model or str(upscale_model) == "None":
                log(f"{self.NODE_NAME}: Use Upscale Model is on but no model is selected. "
                    f"Falling back to a plain {scaling_method} resize.", message_type="warning")
            else:
                mid_width, mid_height = model_intermediate_size(
                    og_width, og_height, int(max_longest_length),
                    None if str(model_multiple) == "None" else int(model_multiple),
                )
                if mid_width <= og_width and mid_height <= og_height:
                    log(
                        f"{self.NODE_NAME}: {og_width}x{og_height} already fills the "
                        f"{int(max_longest_length)} ceiling, so there's no room for a model pass. "
                        f"Resizing straight to {target_width}x{target_height}.",
                        message_type="warning",
                    )
                else:
                    log(f"{self.NODE_NAME}: Model upscale {og_width}x{og_height} -> "
                        f"{mid_width}x{mid_height} (cap {int(max_longest_length)}), "
                        f"then down to {target_width}x{target_height}.")
                    model = self._load_model(upscale_model)
                    source_batch = torch.cat(unpacked_images, dim=0)
                    upscaled = self._model_upscale_to(
                        model, source_batch, mid_width, mid_height, int(max_longest_length))
                    unpacked_images = [upscaled[i:i + 1] for i in range(upscaled.shape[0])]
                    model_used = True

                    # "Unless the user actually wants it to be 2304x4032, in
                    # which case we just forget the downscale."
                    if (mid_width, mid_height) == (target_width, target_height):
                        scale = False
                        log(f"{self.NODE_NAME}: The intermediate is already the target — skipping the downscale.")

                    # Masks were sized to the source; the images have moved on.
                    if unpacked_masks and not scale:
                        log(f"{self.NODE_NAME}: Resizing the mask to match the model output.")
                        unpacked_masks = [
                            image2mask(
                                fit_resize_image(
                                    tensor2pil(m).convert("L"), mid_width, mid_height,
                                    fit, resize_sampler).convert("L")
                            )
                            for m in unpacked_masks
                        ]

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

        # Trust the tensor over the plan from here on. The model path can leave
        # the batch at the intermediate size when the downscale is skipped, and
        # a latent built from stale numbers would silently mismatch the image.
        out_height, out_width = image_batch.shape[1], image_batch.shape[2]

        # ---------------- latent ----------------
        if vae is None:
            latent_info, latent = generate_latent_image_data(
                width      = out_width,
                height     = out_height,
                batch_size = image_batch.shape[0],
                model_type = base_model,
            )
            log(f"{self.NODE_NAME}: No VAE connected. Generated an {latent_info} latent of {out_width}x{out_height}.")
        elif mask_batch is None:
            # Straight encode, but through the shared tiling helper rather than
            # generate_latent_image_data: the model path can hand us a 4096x4096
            # batch, which is exactly where a single-pass encode falls over.
            # encode_auto measures free VRAM and only tiles when it has to,
            # which matters because tiling a 2D VAE costs ~3x either way.
            samples, tile_px = encode_auto(vae, image_batch)
            latent = {"samples": samples}
            latent_info = "encoded"
            log(f"{self.NODE_NAME}: Found a VAE but no mask, encoding the image batch into a latent "
                f"({'tiled at %dpx' % tile_px if tile_px else 'single pass'}).")
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
                width           = out_width,
                height          = out_height,
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
            f"at {out_width}x{out_height}"
            f"{' via an upscale model' if model_used else ''}.",
            message_type="finish",
        )
        return (image_batch, mask_batch, latent, out_width, out_height)
