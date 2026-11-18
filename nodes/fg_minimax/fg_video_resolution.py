"""Device-agnostic video nodes: super resolution, SDR to HDR, frame generation.

Replaces the NVIDIA Video Effects (nvvfx) versions. Every model here is plain
PyTorch, so the nodes run on whatever device ComfyUI selected (Intel XPU,
CUDA, or the processor).

Models are never downloaded by this file.
    Super resolution   picks a file from models/upscale_models and loads it itself.
    Frame generation   picks a file from models/frame_interpolation and loads it itself.
"""

import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

import comfy.utils
import folder_paths
from comfy import model_management
from comfy_api.latest import IO

# spandrel ships with ComfyUI, but the node pack should still load without it.
try:
    from spandrel import ModelLoader, ImageModelDescriptor
    SPANDREL_AVAILABLE = True
except Exception:
    ModelLoader = ImageModelDescriptor = None
    SPANDREL_AVAILABLE = False


# The RIFE and FILM architectures live in ComfyUI core. Older ComfyUI builds
# don't have them; the rest of this file should still load there.
try:
    from comfy_extras.frame_interpolation_models.ifnet import IFNet, detect_rife_config
    from comfy_extras.frame_interpolation_models.film_net import FILMNet
    FRAME_INTERPOLATION_AVAILABLE = True
except Exception:
    IFNet = detect_rife_config = FILMNet = None
    FRAME_INTERPOLATION_AVAILABLE = False


def list_frame_interpolation_models() -> list[str]:
    try:
        return folder_paths.get_filename_list("frame_interpolation") or ["None"]
    except Exception:
        return ["None"]


MAX_BATCH_PIXELS = 1024 * 1024 * 16

RESIZE_SCALE_BY = "scale by multiplier"
RESIZE_TARGET_DIMENSIONS = "target dimensions"

GENERATION_MULTIPLY_FRAME_RATE = "multiply frame rate"
GENERATION_SINGLE_FRAME = "one new frame at a chosen position"

SHOT_CHANGE_DETECTION_ON = "on"
SHOT_CHANGE_DETECTION_OFF = "off"

NODE_CATEGORY = "Farrenzo's Garbage/Video"

HDR_COLORSPACE_HLG = "HDR"
HDR_COLORSPACE_PQ = "HDR PQ"

PQ_M1 = 2610 / 16384
PQ_M2 = 2523 / 4096 * 128
PQ_C1 = 3424 / 4096
PQ_C2 = 2413 / 4096 * 32
PQ_C3 = 2392 / 4096 * 32

HLG_A = 0.17883277
HLG_B = 1 - 4 * HLG_A
HLG_C = 0.5 - HLG_A * math.log(4 * HLG_A)

SCENE_MIDDLE_GRAY = 0.18

# Linear BT.709 primaries -> linear BT.2020 primaries (ITU-R BT.2087).
BT709_TO_BT2020 = (
    (0.6274, 0.3293, 0.0433),
    (0.0691, 0.9195, 0.0114),
    (0.0164, 0.0880, 0.8956),
)
BT2020_LUMINANCE_WEIGHTS = (0.2627, 0.6780, 0.0593)


# Upscale models are reused across runs, so the loaded one is kept at module
# scope. One at a time: switching models drops the previous one.
_UPSCALE_MODEL_CACHE = {}


def load_upscale_model(model_name: str):
    """Turn a file name from models/upscale_models into a loaded Spandrel model."""
    if not model_name or str(model_name) == "None":
        raise ValueError("Video Super Resolution: no upscale model is selected. Put one in models/upscale_models.")
    if not SPANDREL_AVAILABLE:
        raise RuntimeError("Video Super Resolution: spandrel isn't importable, so upscale models can't be loaded.")

    model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
    cached = _UPSCALE_MODEL_CACHE.get(model_path)
    if cached is not None:
        return cached

    state_dict = comfy.utils.load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in state_dict:
        state_dict = comfy.utils.state_dict_prefix_replace(state_dict, {"module.": ""})
    model = ModelLoader().load_from_state_dict(state_dict).eval()

    if not isinstance(model, ImageModelDescriptor):
        raise ValueError(f"Video Super Resolution: '{model_name}' is not a single-image upscale model.")

    _UPSCALE_MODEL_CACHE.clear()
    _UPSCALE_MODEL_CACHE[model_path] = model
    return model


_FRAME_INTERPOLATION_MODEL_CACHE = {}


def load_frame_interpolation_model(model_name: str) -> torch.nn.Module:
    """Turn a file name from models/frame_interpolation into a loaded RIFE or FILM model."""
    if not model_name or str(model_name) == "None":
        raise ValueError("Video Frame Generation: no model is selected. Put a RIFE or FILM file in models/frame_interpolation.")
    if not FRAME_INTERPOLATION_AVAILABLE:
        raise RuntimeError(
            "Video Frame Generation: this ComfyUI build has no built-in frame interpolation models. Update ComfyUI."
        )

    model_path = folder_paths.get_full_path_or_raise("frame_interpolation", model_name)
    cached = _FRAME_INTERPOLATION_MODEL_CACHE.get(model_path)
    if cached is not None:
        return cached

    state_dict = comfy.utils.load_torch_file(model_path, safe_load=True)

    if "extract.extract_sublevels.convs.0.0.conv.weight" in state_dict:
        model = FILMNet()
    else:
        # Raw RIFE checkpoints name their blocks block0..block4; ComfyUI's IFNet expects blocks.0..blocks.4.
        state_dict = comfy.utils.state_dict_prefix_replace(state_dict, {"module.": "", "flownet.": ""})
        renamed = {}
        for key, value in state_dict.items():
            if key.startswith(("teacher.", "caltime.")):
                continue
            for block_index in range(5):
                prefix = f"block{block_index}."
                if key.startswith(prefix):
                    key = f"blocks.{block_index}.{key[len(prefix):]}"
                    break
            renamed[key] = value
        state_dict = renamed
        try:
            head_channels, block_channels = detect_rife_config(state_dict)
        except (KeyError, ValueError):
            raise ValueError(
                f"Video Frame Generation: '{model_name}' is not a supported RIFE or FILM file. "
                f"Older RIFE releases such as rife49.pth use a different layout; rife426.pth works."
            )
        model = IFNet(head_ch=head_channels, channels=block_channels)

    model.load_state_dict(state_dict)
    use_half_precision = model_management.should_use_fp16(model_management.get_torch_device())
    model.eval().to(torch.float16 if use_half_precision else torch.float32)

    _FRAME_INTERPOLATION_MODEL_CACHE.clear()
    _FRAME_INTERPOLATION_MODEL_CACHE[model_path] = model
    return model


def split_alpha(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    alpha = images[..., 3:4] if images.shape[-1] > 3 else None
    return images[..., :3], alpha


# --------------------------------------------------------------------------
# Super resolution
# --------------------------------------------------------------------------

class FG_VideoSuperResolution(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="FG_VideoSuperResolution",
            display_name="Video Super Resolution",
            category=NODE_CATEGORY,
            description="Upscale, denoise, or deblur video frames with any Spandrel upscale model, on any device.",
            search_aliases=["upscale", "super resolution", "vsr", "denoise", "deblur", "video upscale"],
            inputs=[
                IO.Image.Input("images", tooltip="The video frames to upscale. An alpha channel is resized alongside and kept."),
                IO.Combo.Input("upscale_model", options=folder_paths.get_filename_list("upscale_models") or ["None"],
                               tooltip="🧠 The model that does the upscaling, from models/upscale_models. Its native scale "
                                       "(2x, 4x) does not have to match the size you ask for. A 1x denoise or deblur model "
                                       "with scale 1.0 keeps the input dimensions."),
                IO.DynamicCombo.Input("resize_type", tooltip="How the output size is chosen. scale by multiplier: input size "
                                                             "times a number. target dimensions: an exact width and height.",
                                      options=[
                    IO.DynamicCombo.Option(RESIZE_SCALE_BY, [
                        IO.Float.Input("scale", default=2.0, min=1.0, max=8.0, step=0.01,
                                       tooltip="Output size as a multiple of the input. 2.0 turns 1280x720 into 2560x1440."),
                    ]),
                    IO.DynamicCombo.Option(RESIZE_TARGET_DIMENSIONS, [
                        IO.Int.Input("width", default=1920, min=64, max=8192, step=2,
                                     tooltip="Output width in pixels. The frame is stretched to fit, so keep the input's "
                                             "aspect ratio: 1280x720 -> 1920x1080, not 1920x1200."),
                        IO.Int.Input("height", default=1080, min=64, max=8192, step=2,
                                     tooltip="Output height in pixels."),
                    ]),
                ]),
                IO.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Blend between a plain bicubic resize (0.0) and the model output (1.0). "
                                       "0.7 softens a model that over-sharpens."),
                IO.Int.Input("tile_size", default=1024, min=128, max=4096, step=64,
                             tooltip="Frames larger than this are processed in overlapping tiles. Larger is faster but needs "
                                     "more memory. Halved automatically when memory runs out."),
            ],
            outputs=[
                IO.Image.Output("upscaled_images", tooltip="The frames at the requested size, same frame count as the input."),
            ],
        )

    @classmethod
    def execute(cls, images, upscale_model, resize_type, strength, tile_size) -> IO.NodeOutput:
        images, alpha = split_alpha(images)
        frame_count, input_height, input_width, _ = images.shape

        # A dynamic combo arrives as a dict: its own selection plus only the inputs of that selection.
        selected_resize_type = resize_type["resize_type"]
        if selected_resize_type == RESIZE_SCALE_BY:
            output_width = int(input_width * resize_type["scale"])
            output_height = int(input_height * resize_type["scale"])
        elif selected_resize_type == RESIZE_TARGET_DIMENSIONS:
            output_width = resize_type["width"]
            output_height = resize_type["height"]
        else:
            raise ValueError(f"Unsupported resize type: {selected_resize_type}")

        # Even dimensions keep 4:2:0 video encoders happy.
        output_width = max(2, round(output_width / 2) * 2)
        output_height = max(2, round(output_height / 2) * 2)

        # upscale_model arrives as a file name from the dropdown, not a loaded model.
        model = load_upscale_model(upscale_model)
        model_scale = model.scale
        device = model_management.get_torch_device()
        memory_required = model_management.module_size(model.model)
        memory_required += (tile_size * tile_size * 3) * 4 * max(model_scale, 1.0) * 384.0
        model_management.free_memory(memory_required, device)

        output_device = model_management.intermediate_device()
        output_dtype = model_management.intermediate_dtype()
        output = torch.empty((frame_count, output_height, output_width, 3), device=output_device, dtype=output_dtype)

        overlap = 32
        tile = tile_size
        progress_bar = comfy.utils.ProgressBar(frame_count)

        # The model only occupies the device while this node runs.
        model.to(device)
        try:
            for frame_index in tqdm(range(frame_count), desc="Video super resolution"):
                frame = images[frame_index:frame_index + 1].movedim(-1, 1).to(device=device, dtype=torch.float32)

                while True:
                    try:
                        upscaled = comfy.utils.tiled_scale(
                            frame,
                            lambda tile_pixels: model(tile_pixels.float()),
                            tile_x=tile,
                            tile_y=tile,
                            overlap=overlap,
                            upscale_amount=model_scale,
                            output_device=output_device,
                        )
                        break
                    except Exception as error:
                        model_management.raise_non_oom(error)
                        tile //= 2
                        if tile < 128:
                            raise error
                        model_management.soft_empty_cache()

                upscaled = upscaled.float().clamp(0.0, 1.0)
                if upscaled.shape[2] != output_height or upscaled.shape[3] != output_width:
                    # A 4x model asked for 2x has to be shrunk: area averaging when shrinking, bicubic when enlarging.
                    shrinking = upscaled.shape[2] >= output_height and upscaled.shape[3] >= output_width
                    upscaled = comfy.utils.common_upscale(
                        upscaled, output_width, output_height, "area" if shrinking else "bicubic", "disabled",
                    ).clamp(0.0, 1.0)

                if strength < 1.0:
                    plain_resize = comfy.utils.common_upscale(
                        frame.to(output_device), output_width, output_height, "bicubic", "disabled",
                    )
                    upscaled = torch.lerp(plain_resize.float(), upscaled, strength)

                output[frame_index].copy_(upscaled[0].movedim(0, -1).clamp(0.0, 1.0))
                progress_bar.update(1)
        finally:
            model.to("cpu")
            model_management.soft_empty_cache()

        if alpha is not None:
            upscaled_alpha = F.interpolate(
                alpha.movedim(-1, 1).float(),
                size=(output_height, output_width),
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1)
            output = torch.cat((output, upscaled_alpha.to(output)), dim=-1)

        return IO.NodeOutput(output)


# --------------------------------------------------------------------------
# SDR to HDR (analytic inverse tone mapping, no model)
# --------------------------------------------------------------------------

def srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    image = image.clamp(0.0, 1.0)
    return torch.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)


def display_light_to_pq(display_light: torch.Tensor) -> torch.Tensor:
    """Display light in nits -> BT.2100 PQ signal in [0, 1]."""
    normalized = (display_light / 10000.0).clamp(0.0, 1.0) ** PQ_M1
    return ((PQ_C1 + PQ_C2 * normalized) / (1.0 + PQ_C3 * normalized)) ** PQ_M2


def display_light_to_hlg(display_light: torch.Tensor, peak_luminance: float) -> torch.Tensor:
    """Display light in nits -> BT.2100 HLG signal in [0, 1], for a display of the given peak."""
    display_light = display_light.clamp(0.0, peak_luminance)
    weights = display_light.new_tensor(BT2020_LUMINANCE_WEIGHTS)
    display_luminance = (display_light * weights).sum(dim=-1, keepdim=True)

    system_gamma = 1.2 + 0.42 * math.log10(peak_luminance / 1000.0)
    luminance_ratio = (display_luminance / peak_luminance).clamp_min(torch.finfo(display_light.dtype).eps)
    scene_light = (display_light / peak_luminance) * luminance_ratio.pow((1.0 - system_gamma) / system_gamma)
    scene_light.clamp_(0.0, 1.0)

    low = torch.sqrt(3.0 * scene_light)
    high = HLG_A * torch.log((12.0 * scene_light - HLG_B).clamp_min(torch.finfo(scene_light.dtype).eps)) + HLG_C
    return torch.where(scene_light <= 1.0 / 12.0, low, high)


def expand_sdr_to_display_light(
    image: torch.Tensor,
    contrast: int,
    saturation: int,
    middle_gray: int,
    luminance: int,
    highlight_rolloff_exponent: float,
) -> torch.Tensor:
    """sRGB frames in [0, 1] -> BT.2020 display light in nits.

    Mid-tones are placed at `middle_gray` nits and stay there; only the
    brightest part of the range is stretched up to `luminance` nits.
    """
    linear = srgb_to_linear(image)
    matrix = linear.new_tensor(BT709_TO_BT2020)
    linear = (linear @ matrix.T).clamp_min(0.0)

    weights = linear.new_tensor(BT2020_LUMINANCE_WEIGHTS)
    eps = torch.finfo(linear.dtype).eps
    scene_luminance = (linear * weights).sum(dim=-1, keepdim=True)

    # Saturation: push each channel away from (or toward) the pixel's luminance.
    linear = (scene_luminance + (linear - scene_luminance) * (saturation / 100.0)).clamp_min(0.0)

    # Contrast: power curve pivoting on scene middle grey, so 18% grey never moves.
    contrasted_luminance = SCENE_MIDDLE_GRAY * (scene_luminance / SCENE_MIDDLE_GRAY).clamp_min(eps) ** (contrast / 100.0)

    # Where SDR white lands before highlight expansion. middle_gray=50 -> 278 nits.
    sdr_white_nits = middle_gray / SCENE_MIDDLE_GRAY
    relative_luminance = contrasted_luminance.clamp(0.0, 1.0)

    # Highlight gain rises from 1.0 in the mid-tones to luminance / sdr_white_nits at SDR white.
    highlight_gain = (luminance / sdr_white_nits) ** (relative_luminance ** highlight_rolloff_exponent)
    display_luminance = sdr_white_nits * relative_luminance * highlight_gain

    display_light = linear * (display_luminance / scene_luminance.clamp_min(eps))
    return display_light.clamp_(0.0, float(luminance))


class FG_VideoHDR(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="FG_VideoHDR",
            display_name="Video SDR to HDR",
            category=NODE_CATEGORY,
            description="Convert SDR frames to HLG or PQ HDR image data with an analytic highlight expansion, on any device.",
            search_aliases=["hdr", "video hdr", "hdr10", "hlg", "pq", "sdr to hdr", "inverse tone mapping"],
            inputs=[
                IO.Image.Input("images", tooltip="Standard dynamic range frames (ordinary sRGB). An alpha channel is passed through."),
                IO.Int.Input("contrast", default=100, min=0, max=200,
                             tooltip="100 leaves contrast unchanged. Pivots on middle grey, so mid-tone brightness does not "
                                     "move. 120 deepens shadows and lifts highlights."),
                IO.Int.Input("saturation", default=100, min=0, max=200,
                             tooltip="100 leaves saturation unchanged. 0 is greyscale, 120 is noticeably more vivid."),
                IO.Int.Input("middle_gray", default=50, min=10, max=100,
                             tooltip="Brightness in nits given to 18% grey, which sets how bright the whole picture feels. "
                                     "50 places SDR white at 278 nits before highlight expansion; 30 gives a dimmer, more "
                                     "cinematic picture."),
                IO.Int.Input("luminance", default=650, min=400, max=2000,
                             tooltip="Peak brightness in nits of the display you are targeting. The brightest SDR white is "
                                     "stretched up to this value. 650 suits most HDR monitors; 1000 or more suits HDR televisions."),
                IO.Float.Input("highlight_rolloff_exponent", default=3.0, min=1.0, max=8.0, step=0.1,
                               tooltip="How far into the highlights the expansion is confined. 1.0 brightens most of the "
                                       "image; 6.0 only lifts speculars, lamps and sky."),
                IO.Boolean.Input("debanding", default=False,
                                 tooltip="Adds one 8-bit step of dither noise before expansion to hide banding in skies and "
                                         "gradients from 8-bit sources. This is dithering, not a learned debander."),
                IO.Combo.Input("output_colorspace", options=[HDR_COLORSPACE_HLG, HDR_COLORSPACE_PQ], default=HDR_COLORSPACE_HLG,
                               tooltip="HDR: BT.2020 with the HLG curve, built for the selected peak luminance. "
                                       "HDR PQ: BT.2020 with the PQ curve, the encoding HDR10 uses."),
            ],
            outputs=[
                IO.Image.Output("hdr_images", tooltip="Normalized RGB values using the selected HDR colorspace. They look "
                                                      "washed out in a normal preview; that is expected."),
            ],
        )

    @classmethod
    def execute(cls, images, contrast, saturation, middle_gray, luminance, highlight_rolloff_exponent, debanding, output_colorspace) -> IO.NodeOutput:
        images, alpha = split_alpha(images)
        frame_count, height, width, _ = images.shape
        device = model_management.get_torch_device()
        output = torch.empty(images.shape, device=images.device, dtype=torch.float32)

        generator = torch.Generator(device="cpu").manual_seed(0)
        batch_size = max(1, MAX_BATCH_PIXELS // (width * height))
        progress_bar = comfy.utils.ProgressBar(frame_count)

        for start in range(0, frame_count, batch_size):
            batch = images[start:start + batch_size]
            if debanding:
                # Triangular noise spanning one 8-bit step, generated on the processor so results match across devices.
                noise = torch.rand(batch.shape, generator=generator) - torch.rand(batch.shape, generator=generator)
                batch = batch + noise.to(batch) / 255.0
            batch = batch.to(device=device, dtype=torch.float32)

            display_light = expand_sdr_to_display_light(
                batch, contrast, saturation, middle_gray, luminance, highlight_rolloff_exponent,
            )
            if output_colorspace == HDR_COLORSPACE_HLG:
                encoded = display_light_to_hlg(display_light, float(luminance))
            elif output_colorspace == HDR_COLORSPACE_PQ:
                encoded = display_light_to_pq(display_light)
            else:
                raise ValueError(f"Unsupported output colorspace: {output_colorspace}")

            output[start:start + batch.shape[0]].copy_(encoded)
            progress_bar.update(batch.shape[0])

        if alpha is not None:
            output = torch.cat((output, alpha.to(output)), dim=-1)

        return IO.NodeOutput(output)


# --------------------------------------------------------------------------
# Frame generation
# --------------------------------------------------------------------------

def is_shot_change(previous_image: torch.Tensor, current_image: torch.Tensor, threshold: float) -> bool:
    """Mean absolute difference between two frames (height, width, channels) on a small copy."""
    def shrink(image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(image.movedim(-1, 0).unsqueeze(0).float(), size=(64, 64), mode="area")

    return (shrink(previous_image) - shrink(current_image)).abs().mean().item() > threshold


class FG_VideoFrameGeneration(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="FG_VideoFrameGeneration",
            display_name="Video Frame Generation",
            category=NODE_CATEGORY,
            description="Generate intermediate video frames with a RIFE or FILM model, on any device.",
            search_aliases=["frame generation", "interpolation", "slow motion", "vfi", "rife", "film", "frame rate"],
            inputs=[
                IO.Image.Input("images", tooltip="The video frames, in order. At least 2 are needed. An alpha channel is dropped."),
                IO.Combo.Input("frame_interpolation_model", options=list_frame_interpolation_models(),
                               tooltip="🧠 The RIFE or FILM model that generates the in-between frames, from "
                                       "models/frame_interpolation. rife426.pth works; older files such as rife49.pth do not."),
                IO.DynamicCombo.Input("frame_generation_mode",
                                      tooltip="How many new frames go between each pair of original frames. "
                                              "multiply frame rate: the usual choice. Fills each gap evenly, so 24 frames "
                                              "per second becomes 48 or 72. "
                                              "one new frame at a chosen position: adds exactly one frame per gap, placed "
                                              "wherever you say between the two originals.",
                                      options=[
                    IO.DynamicCombo.Option(GENERATION_MULTIPLY_FRAME_RATE, [
                        IO.Int.Input("multiplier", default=2, min=2, max=16,
                                     tooltip="How many times smoother the result is. 2 adds 1 new frame per gap (24 -> 48 "
                                             "frames per second), 4 adds 3 (24 -> 96). Save at the multiplied frame rate "
                                             "for smooth motion, or at the original rate for slow motion."),
                    ]),
                    IO.DynamicCombo.Option(GENERATION_SINGLE_FRAME, [
                        IO.Float.Input("position_between_frames", default=0.5, min=0.01, max=0.99, step=0.01,
                                       tooltip="Where the new frame sits between each original pair. 0.5 is exactly halfway "
                                               "(the same result as multiplier 2). 0.25 is a quarter of the way from the "
                                               "first frame to the second; 0.9 is almost the second frame."),
                    ]),
                ]),
                IO.DynamicCombo.Input("shot_change_detection",
                                      tooltip="What to do when two neighbouring frames belong to different shots. "
                                              "on: across a cut, repeat the nearest real frame instead of morphing two "
                                              "unrelated shots into a smear. off: always generate, even across cuts.",
                                      options=[
                    IO.DynamicCombo.Option(SHOT_CHANGE_DETECTION_ON, [
                        IO.Float.Input("shot_change_threshold", default=0.20, min=0.01, max=1.0, step=0.01,
                                       tooltip="How different two neighbouring frames must be (average pixel difference, "
                                               "0 to 1) to count as a cut. Ordinary motion scores about 0.01 to 0.05; a hard "
                                               "cut scores 0.3 or more. Lower it if cuts are being morphed; raise it if "
                                               "fast action is being treated as a cut."),
                    ]),
                    IO.DynamicCombo.Option(SHOT_CHANGE_DETECTION_OFF, []),
                ]),
            ],
            outputs=[
                IO.Image.Output("interpolated_images", tooltip="Original and generated frames interleaved. 25 input frames "
                                                               "with multiplier 2 give 49."),
            ],
        )

    @classmethod
    def execute(cls, images, frame_interpolation_model, frame_generation_mode, shot_change_detection) -> IO.NodeOutput:
        images = images[..., :3]
        frame_count, height, width, _ = images.shape
        if frame_count < 2:
            return IO.NodeOutput(images)

        # A dynamic combo arrives as a dict: its own selection plus only the inputs of that selection.
        selected_mode = frame_generation_mode["frame_generation_mode"]
        if selected_mode == GENERATION_MULTIPLY_FRAME_RATE:
            multiplier = frame_generation_mode["multiplier"]
            timestep_values = [index / multiplier for index in range(1, multiplier)]
        elif selected_mode == GENERATION_SINGLE_FRAME:
            timestep_values = [float(frame_generation_mode["position_between_frames"])]
        else:
            raise ValueError(f"Unsupported frame generation mode: {selected_mode}")

        automatic_shot_change_detection = shot_change_detection["shot_change_detection"] == SHOT_CHANGE_DETECTION_ON
        shot_change_threshold = shot_change_detection.get("shot_change_threshold", 0.20)
        generated_per_pair = len(timestep_values)

        # frame_interpolation_model arrives as a file name from the dropdown, not a loaded model.
        model = load_frame_interpolation_model(frame_interpolation_model)
        device = model_management.get_torch_device()
        dtype = next(model.parameters()).dtype
        memory_required = model_management.module_size(model) + model.memory_used_forward(images.shape, dtype)
        model_management.free_memory(memory_required, device)
        pad_align = getattr(model, "pad_align", 1)

        def prepare_frame(index: int) -> torch.Tensor:
            frame = images[index:index + 1].movedim(-1, 1).to(device=device, dtype=dtype)
            if pad_align > 1:
                pad_bottom = (pad_align - height % pad_align) % pad_align
                pad_right = (pad_align - width % pad_align) % pad_align
                if pad_bottom or pad_right:
                    frame = F.pad(frame.float(), (0, pad_right, 0, pad_bottom), mode="reflect").to(dtype)
            return frame

        output_device = model_management.intermediate_device()
        output_dtype = model_management.intermediate_dtype()
        pair_count = frame_count - 1
        output_count = pair_count * (generated_per_pair + 1) + 1
        output = torch.empty((output_count, height, width, 3), device=output_device, dtype=output_dtype)
        output[0].copy_(images[0])
        output_index = 1

        # The model only occupies the device while this node runs.
        model.to(device)
        try:
            previous_frame = prepare_frame(0)
            padded_height, padded_width = previous_frame.shape[2], previous_frame.shape[3]
            timestep_planes = torch.tensor(timestep_values, device=device, dtype=dtype).reshape(-1, 1, 1, 1)
            timestep_planes = timestep_planes.expand(-1, 1, padded_height, padded_width)

            multi_timestep_forward = getattr(model, "forward_multi_timestep", None)
            batch_size = generated_per_pair
            feature_cache = {}
            progress_bar = comfy.utils.ProgressBar(pair_count)

            for pair_index in tqdm(range(pair_count), desc="Video frame generation"):
                current_frame = prepare_frame(pair_index + 1)
                current_image = images[pair_index + 1]

                if automatic_shot_change_detection and is_shot_change(images[pair_index], current_image, shot_change_threshold):
                    for timestep_value in timestep_values:
                        nearest_image = images[pair_index] if timestep_value < 0.5 else current_image
                        output[output_index].copy_(nearest_image)
                        output_index += 1
                    feature_cache.pop("next", None)
                else:
                    # The second frame of this pair is the first frame of the next one, so its features are reused.
                    feature_cache["img0"] = feature_cache.pop("next") if "next" in feature_cache else model.extract_features(previous_frame)
                    feature_cache["img1"] = model.extract_features(current_frame)
                    feature_cache["next"] = feature_cache["img1"]

                    generated = None
                    if multi_timestep_forward is not None:
                        try:
                            generated = multi_timestep_forward(previous_frame, current_frame, timestep_values, cache=feature_cache)
                        except model_management.OOM_EXCEPTION:
                            model_management.soft_empty_cache()
                            multi_timestep_forward = None

                    if generated is not None:
                        output[output_index:output_index + generated_per_pair].copy_(
                            generated[:, :, :height, :width].movedim(1, -1).clamp(0.0, 1.0)
                        )
                        output_index += generated_per_pair
                    else:
                        done = 0
                        while done < generated_per_pair:
                            count = min(batch_size, generated_per_pair - done)
                            try:
                                generated = model(
                                    previous_frame.expand(count, -1, -1, -1),
                                    current_frame.expand(count, -1, -1, -1),
                                    timestep=timestep_planes[done:done + count],
                                    cache=feature_cache,
                                )
                            except model_management.OOM_EXCEPTION:
                                if batch_size <= 1:
                                    raise
                                batch_size = max(1, batch_size // 2)
                                model_management.soft_empty_cache()
                                continue
                            output[output_index:output_index + count].copy_(
                                generated[:, :, :height, :width].movedim(1, -1).clamp(0.0, 1.0)
                            )
                            output_index += count
                            done += count

                output[output_index].copy_(current_image)
                output_index += 1
                previous_frame = current_frame
                progress_bar.update(1)
        finally:
            model.to("cpu")
            feature_cache = None
            if hasattr(model, "_warp_grids"):
                # Plain tensors cached on the device; .to("cpu") does not move them.
                model._warp_grids = {}
            model_management.soft_empty_cache()

        return IO.NodeOutput(output)
