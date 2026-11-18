"""
FG_MiniMaxH3PaintLatent - paint still images into the video part of a MiniMax H3
audio-video latent and mark them as "keep" in a nested noise mask.

Why this exists
  MiniMaxH3AddGuide only adds conditioning tokens (the latent stays an empty canvas).
  The MiniMaxH3 model class in comfy/model_base.py ALSO supports a pre-painted latent:
  - extra_conds() forwards the sampler's denoise_mask to the model (0 = keep, 1 = regenerate),
  - scale_latent_inpaint() injects the kept latent frames nearly clean.
  No stock node builds the nested latent plus nested mask, so this node does.

Placement rule
  Latent frames cover FRAME_PER_TOKEN = (1, 4, 4, 4, 4) pixel frames, repeating. A still image
  encodes to exactly one latent frame, so it fits the single-pixel-frame slots (pixel frames
  0, 17, 34, ... 153 for a 158 frame video). With snap_to_single_frame_slots on, each requested
  pixel frame moves to the nearest such slot.
"""

import bisect

import torch

import comfy.nested_tensor
import comfy.utils
from comfy.ldm.minimax.model import FRAME_PER_TOKEN


def pixel_frame_starts(latent_frame_count):
    """First pixel frame of every latent frame, plus the end of the last one."""
    starts = [0]
    for latent_index in range(latent_frame_count):
        starts.append(starts[-1] + FRAME_PER_TOKEN[latent_index % 5])
    return starts


def resolve_placements(requested_pixel_frames, latent_frame_count, snap_to_single_frame_slots):
    """Return a list of (pixel_frame, latent_index, slot_width_in_pixel_frames).
    Negative requests count from the end of the video."""
    starts = pixel_frame_starts(latent_frame_count)
    total_pixel_frames = starts[-1]
    single_frame_slot_starts = [starts[index] for index in range(latent_frame_count)
                                if starts[index + 1] - starts[index] == 1]
    placements, used_latent_indices = [], set()
    for requested in requested_pixel_frames:
        pixel_frame = requested if requested >= 0 else total_pixel_frames + requested
        if not 0 <= pixel_frame < total_pixel_frames:
            raise ValueError(f"pixel frame {requested} is outside the video's {total_pixel_frames} frames")
        if snap_to_single_frame_slots:
            pixel_frame = min(single_frame_slot_starts, key=lambda slot_start: abs(slot_start - pixel_frame))
        latent_index = bisect.bisect_right(starts, pixel_frame) - 1
        if latent_index in used_latent_indices:
            raise ValueError(f"two images land on latent frame {latent_index}; spread the pixel frames further apart")
        used_latent_indices.add(latent_index)
        placements.append((pixel_frame, latent_index, starts[latent_index + 1] - starts[latent_index]))
    return placements


class FG_MiniMaxH3PaintLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Empty MiniMax H3 audio-video latent (from the H3 conditioning node)."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "keyframe_images": ("IMAGE", {"tooltip": "Batch of still images, one per pixel frame index below, in the same order."}),
                "pixel_frame_indices": ("STRING", {"default": "0, 51, 102, 153",
                                                   "tooltip": "Comma separated pixel frame for each image. Negative counts from the end."}),
                "snap_to_single_frame_slots": ("BOOLEAN", {"default": True,
                                                           "tooltip": "Move each index to the nearest pixel frame whose latent frame covers exactly one pixel frame (multiples of 17)."}),
                "keep_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "1.0 pins the painted frame completely. Lower values let the model change it (mask = 1 - keep_strength)."}),
                "resize_mode": (["center", "disabled"], {"default": "center",
                                                         "tooltip": "center = cover-crop keeping aspect ratio, disabled = stretch to the canvas."}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "placement_report")
    FUNCTION = "paint"
    CATEGORY = "FG/MiniMaxH3"

    def paint(self, latent, vae, keyframe_images, pixel_frame_indices,
              snap_to_single_frame_slots, keep_strength, resize_mode):
        samples = latent["samples"]
        if (not samples.is_nested or len(samples.tensors) != 2
                or samples.tensors[0].ndim != 5 or samples.tensors[0].shape[1] != 24):
            raise ValueError("FG_MiniMaxH3PaintLatent expects a MiniMax H3 audio-video latent")
        video_latent, audio_latent = samples.tensors
        canvas_height = video_latent.shape[3] * 16
        canvas_width = video_latent.shape[4] * 16

        requested = [int(token) for token in pixel_frame_indices.replace(";", ",").split(",") if token.strip()]
        if len(requested) != keyframe_images.shape[0]:
            raise ValueError(f"{keyframe_images.shape[0]} images but {len(requested)} pixel frame indices")
        placements = resolve_placements(requested, video_latent.shape[2], snap_to_single_frame_slots)

        painted_video = video_latent.clone()
        video_mask = torch.ones_like(video_latent)   # 1 = regenerate
        audio_mask = torch.ones_like(audio_latent)   # audio is fully regenerated
        report_lines = []
        for image_position, (pixel_frame, latent_index, slot_width) in enumerate(placements):
            image = keyframe_images[image_position:image_position + 1][..., :3]
            resized = comfy.utils.common_upscale(image.movedim(-1, 1), canvas_width, canvas_height,
                                                 "lanczos", resize_mode).movedim(1, -1)
            encoded = vae.encode(resized)
            if (encoded.ndim != 5 or encoded.shape[1] != 24 or encoded.shape[2] != 1
                    or tuple(encoded.shape[3:]) != tuple(video_latent.shape[3:])):
                raise ValueError(f"unexpected encoded still shape {tuple(encoded.shape)}")
            painted_video[:, :, latent_index:latent_index + 1] = encoded.to(
                device=painted_video.device, dtype=painted_video.dtype)
            video_mask[:, :, latent_index:latent_index + 1] = 1.0 - keep_strength
            note = "" if slot_width == 1 else f"  (WARNING: slot covers {slot_width} pixel frames, the still will be stretched)"
            report_lines.append(f"image {image_position}: requested {requested[image_position]} -> pixel frame "
                                f"{pixel_frame} -> latent frame {latent_index}{note}")

        output = latent.copy()
        output["samples"] = comfy.nested_tensor.NestedTensor((painted_video, audio_latent))
        output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
        return (output, "\n".join(report_lines))


