"""
MiniMax H3 Video - unified conditioner
======================================

Merges ComfyUI core's MiniMaxH3ImageToVideo (t2va / fl2va) and
MiniMaxH3ReferenceToVideo (ref2va) into one node covering:

  1. Text to video          - prompt only
  2. Image to video         - ref_image_N
  3. First/last frame       - first_frame (+ optional last_frame)
  4. Video to video         - ref_video_N (+ optional ref_video_audio_N)
  5. Audio to video         - ref_audio_N (lip sync; needs audio_vae)
  6. All of the above       - storyboard + keyframes + dialogue + score

WHY THIS CANNOT BE A NAIVE CONCATENATION OF THE TWO NODES
---------------------------------------------------------
MiniMaxH3Tokenizer.tokenize_with_weights branches:

    if minimax_ref_items:
        ...
    else:
        for i, img in enumerate(images):
            ...

`images=` is silently ignored whenever minimax_ref_items is non-empty, and
calling tokenize twice just discards the first result. So keyframes are
presented through minimax_ref_items as well -- both tokenizer branches emit
identical "<Picture i>: " + vision markup for images, so this is lossless.

The DiT is fine with both at once: PackedLayout takes keyframes= and refs= as
independent arguments and packs keyframe cond rows after the text, then ref
rows after those. model_base.extra_conds reads minimax_keyframes,
minimax_frame_count and minimax_refs as separate kwargs into one payload.

<Picture i> / <Audio j> / <Video k> ORDINALS
--------------------------------------------
The tokenizer counts only the items it actually receives, so ordinals are
positional, NOT socket indices. Leaving a gap in the autogrow slots renumbers
everything after it, and adding a reference video with a soundtrack pushes
standalone audio ordinals up (a soundtrack's <Audio j> is emitted immediately
before its <Video k>). This node logs the resolved map on every run -- read it
before writing prompts that name specific tags.
"""

import logging
import math

import torch
import torchaudio

import nodes
import node_helpers

import comfy.utils
import comfy.nested_tensor
import comfy.model_management
from comfy_api.latest import io


LOG_PREFIX = "[FG MiniMax H3]"

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40

MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3

# Autogrow arrived relatively recently. Without it the node still loads, just
# with a fixed set of reference sockets instead of growing ones.
HAS_AUTOGROW = hasattr(io, "Autogrow")

_FALLBACK_REF_IMAGES = 3
_FALLBACK_REF_VIDEOS = 1
_FALLBACK_REF_AUDIOS = 2

_TAG_NAMES = {"image": "Picture", "audio": "Audio", "video": "Video"}


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def adapt_canvas(width, height):
    """768-short-edge canvas with 768*1344 area cap, per-axis round to 32."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def _resize(image, width, height, crop):
    # image [B, H, W, C] -> [B, height, width, 3]
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _slot_index(name):
    """Numeric suffix of an autogrow socket name, for deterministic ordering."""
    try:
        return int(name.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return 0


def _collect(autogrow_value, prefix, kwargs):
    """
    Normalise reference inputs to an ordered [(socket_name, value)] list.

    Handles both shapes: the dict Autogrow delivers, and the loose per-socket
    kwargs of the fixed-slot fallback. Sorting by numeric suffix means socket
    order is what the frontend shows, not whatever order the dict arrived in.
    """
    if autogrow_value:
        items = list(autogrow_value.items())
    else:
        items = [(k, v) for k, v in kwargs.items() if k.startswith(prefix)]

    items = [(k, v) for k, v in items if v is not None]
    items.sort(key=lambda kv: _slot_index(kv[0]))
    return items


def _ref_inputs(max_count, prefix, io_input, tooltip, autogrow_id):
    """Build autogrow inputs when available, otherwise fixed optional slots."""
    if HAS_AUTOGROW:
        return [io.Autogrow.Input(
            autogrow_id, optional=True,
            template=io.Autogrow.TemplatePrefix(
                input=io_input(prefix.rstrip("_"), tooltip=tooltip),
                prefix=prefix, min=0, max=max_count))]
    return [io_input(f"{prefix}{i}", optional=True, tooltip=tooltip)
            for i in range(max_count)]


class FG_MiniMaxH3_Conditioner(io.ComfyNode):
    """Unified t2va / fl2va / ref2va conditioning for MiniMax H3."""

    @classmethod
    def define_schema(cls):
        inputs = [
            io.Clip.Input("clip", tooltip="📒 CLIP: the Qwen3-VL text/vision encoder."),
            io.Vae.Input("vae", tooltip="🔣 Video VAE: encodes keyframes and visual references."),
            io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                tooltip="Refer to references by tag: <Picture i>, <Video k>, <Audio j>. "
                        "The resolved map is logged to the console on every run."),
            io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
            io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
            io.Int.Input("length", default=124, min=5, max=3600, step=17,
                tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid "
                        "(124 = ~5s; trained range is ~124-362, longer is untested)."),
            io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                tooltip="'match' scales each reference (down only, keeping aspect) to the "
                        "generation's pixel area; 'max' uses the 2048px short edge for best "
                        "identity fidelity. Reference tokens ride through every sampling "
                        "step, so 'max' can be several times slower."),

            io.Vae.Input("audio_vae", optional=True,
                tooltip="🔣 Audio VAE: required for any audio reference. Leave unconnected "
                        "for text/image/video-only modes."),
            io.Image.Input("first_frame", optional=True,
                tooltip="🖼️ Anchors frame 0. Stretched to the canvas (no crop)."),
            io.Image.Input("last_frame", optional=True,
                tooltip="🖼️ Anchors the final frame. Aspect-preserving cover-crop. "
                        "Only valid alongside first_frame."),
        ]

        inputs += _ref_inputs(
            MAX_REF_IMAGES, "ref_image_", io.Image.Input,
            "Reference image: a location, a character, a storyboard panel. "
            "Downscaled if larger than the target, never upscaled.",
            "ref_images")
        inputs += _ref_inputs(
            MAX_REF_VIDEOS, "ref_video_", io.Image.Input,
            "Reference video frames at 24 fps (2-15s). Motion/pose guidance.",
            "ref_videos")
        inputs += _ref_inputs(
            MAX_REF_VIDEOS, "ref_video_audio_", io.Audio.Input,
            "Soundtrack of the same-numbered reference video.",
            "ref_video_audios")
        inputs += _ref_inputs(
            MAX_REF_AUDIOS, "ref_audio_", io.Audio.Input,
            "Standalone reference audio: dialogue for lip sync, or music/ambience.",
            "ref_audios")

        inputs += [
            io.Float.Input("visual_cond_noise_aug", default=0.0, min=0.0, max=1.0,
                step=0.01, optional=True,
                tooltip="Noise augmentation on visual condition latents. 0 = off. "
                        "Raise slightly if keyframes and references fight each other."),
            io.Float.Input("audio_cond_noise_aug", default=0.0, min=0.0, max=1.0,
                step=0.01, optional=True,
                tooltip="Noise augmentation on audio condition latents. 0 = off."),
        ]

        return io.Schema(
            node_id="FG_MiniMaxH3_Conditioner",
            display_name="MiniMax H3 Conditioner (Unified)",
            category="Farrenzo's Garbage/video",
            description="Unified <Picture i> / <Video k> / <Audio j> conditioning for "
                        "MiniMax H3. Keyframes and references in one node. Use the same "
                        "tags when prompting.",
            search_aliases=["H3", "minimax", "text encoder", "encode prompt",
                            "reference to video", "first last frame"],
            inputs=inputs,
            outputs=[io.Conditioning.Output(display_name="positive"),
                     io.Latent.Output(display_name="latent")],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_ref_audio(audio_vae, audio):
        waveform = audio["waveform"]  # [B, C, L]
        sr = audio["sample_rate"]
        vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
        if sr != vae_sr:
            waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
        z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
        return z, z.shape[-1]

    @staticmethod
    def _log_tag_map(presentation, labels):
        """
        Print the tag -> socket mapping the tokenizer will actually produce.

        Ordinals come from position in the presentation list, so this is the
        only reliable way to know what "<Picture 3>" refers to in a given run.
        """
        counters = {"image": 0, "audio": 0, "video": 0}
        lines = []
        for item, label in zip(presentation, labels):
            kind = item["type"]
            counters[kind] += 1
            lines.append(f"  <{_TAG_NAMES[kind]} {counters[kind]}>  =  {label}")

        if lines:
            logging.info("%s prompt tag map:\n%s", LOG_PREFIX, "\n".join(lines))
        else:
            logging.info("%s no references; text-to-video.", LOG_PREFIX)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    @classmethod
    def execute(cls, clip, vae, prompt, width, height, length,
                ref_image_size="match", audio_vae=None,
                first_frame=None, last_frame=None,
                visual_cond_noise_aug=0.0, audio_cond_noise_aug=0.0,
                ref_images=None, ref_videos=None,
                ref_video_audios=None, ref_audios=None,
                **kwargs) -> io.NodeOutput:

        latent, frame_count = _empty_av_latent(width, height, length)
        if frame_count != length:
            logging.info("%s length %d snapped up to %d frames (17k+5 grid, %.2fs).",
                         LOG_PREFIX, length, frame_count, frame_count / FPS)

        image_items = _collect(ref_images, "ref_image_", kwargs)
        video_items = _collect(ref_videos, "ref_video_", kwargs)
        soundtracks = dict(_collect(ref_video_audios, "ref_video_audio_", kwargs))
        audio_items = _collect(ref_audios, "ref_audio_", kwargs)

        # "ref_video_" is a prefix of "ref_video_audio_", so the fallback path
        # would otherwise scoop soundtracks into the video list.
        video_items = [(k, v) for k, v in video_items if not k.startswith("ref_video_audio_")]

        if (soundtracks or audio_items) and audio_vae is None:
            raise ValueError(
                f"{LOG_PREFIX} An audio reference is connected but audio_vae is not. "
                f"Connect the MiniMax H3 audio VAE, or disconnect the audio inputs."
            )
        if last_frame is not None and first_frame is None:
            raise ValueError(
                f"{LOG_PREFIX} last_frame requires first_frame. The model only accepts "
                f"keyframe anchors at frame 0 and frame {frame_count - 1}; a lone "
                f"last_frame is ambiguous."
            )

        # --- keyframes (minimax_keyframes payload) -------------------------
        keyframes = []
        keyframe_labels = []
        if first_frame is not None:
            # geometry anchor: plain stretch to canvas
            keyframes.append({"resolved_frame_index": 0,
                              "image": _resize(first_frame[:1], width, height, "disabled")})
            keyframe_labels.append("first_frame")
        if last_frame is not None:
            # follower: aspect-preserving cover-crop
            keyframes.append({"resolved_frame_index": frame_count - 1,
                              "image": _resize(last_frame[:1], width, height, "center")})
            keyframe_labels.append("last_frame")

        # --- references (minimax_refs payload) -----------------------------
        ref_items = []    # tokenizer presentation, in request order
        ref_blocks = []   # DiT payload, same order
        ref_labels = []   # parallel to ref_items, for the tag map

        for name, img in image_items:
            h, w = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                # aspect-preserving scale (down only) to the generation's pixel area
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            z = vae.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_labels.append(name)
            ref_blocks.append({"kind": "image", "latent_h": th // 16,
                               "latent_w": tw // 16, "latent": z})

        for name, video_frames in video_items:
            # index-paired soundtrack: ref_video_audio_N belongs to ref_video_N
            sound_name = "ref_video_audio_" + name.rsplit("_", 1)[-1]
            soundtrack = soundtracks.get(sound_name)

            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError(
                    f"{LOG_PREFIX} {name}: reference videos need at least 5 frames "
                    f"(~0.2s at 24 fps), got {n}."
                )
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            z = vae.encode(frames)

            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None:
                audio_latent, ref_audio_t = cls._encode_ref_audio(audio_vae, soundtrack)
                # the soundtrack gets its own <Audio j> label, emitted before <Video k>
                ref_items.append({"type": "audio"})
                ref_labels.append(sound_name)

            # Qwen sees the video at 2 fps with timestamps
            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            qwen_frames = frames[sample_idx]
            ref_items.append({"type": "video", "data": qwen_frames,
                              "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
            ref_labels.append(name)
            ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                               "latent_t": z.shape[2], "latent_h": ch // 16,
                               "latent_w": cw // 16, "ref_audio_t": ref_audio_t,
                               "latent": z, "audio_latent": audio_latent})

        for name, audio in audio_items:
            audio_latent, ref_audio_t = cls._encode_ref_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_labels.append(name)
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t,
                               "audio_latent": audio_latent})

        # --- single tokenize pass ------------------------------------------
        # Keyframes ride in through minimax_ref_items because `images=` is dead
        # whenever that list is non-empty. Both tokenizer branches emit the same
        # "<Picture i>: " + vision markup, so nothing is lost.
        presentation = [{"type": "image", "data": kf["image"]} for kf in keyframes]
        presentation.extend(ref_items)
        labels = keyframe_labels + ref_labels

        cls._log_tag_map(presentation, labels)

        if presentation:
            tokens = clip.tokenize(prompt, minimax_ref_items=presentation)
        else:
            tokens = clip.tokenize(prompt)
        cond = clip.encode_from_tokens_scheduled(tokens)

        # --- conditioning payloads -----------------------------------------
        values = {}
        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            values["minimax_keyframes"] = keyframes
            values["minimax_frame_count"] = frame_count
        if ref_blocks:
            values["minimax_refs"] = ref_blocks
        if visual_cond_noise_aug > 0.0:
            values["minimax_visual_cond_noise_aug"] = visual_cond_noise_aug
        if audio_cond_noise_aug > 0.0:
            values["minimax_audio_cond_noise_aug"] = audio_cond_noise_aug

        if values:
            cond = node_helpers.conditioning_set_values(cond, values)

        return io.NodeOutput(cond, latent)


if not HAS_AUTOGROW:
    logging.warning(
        "%s io.Autogrow is unavailable in this ComfyUI build; falling back to "
        "%d fixed reference image slots. Update ComfyUI for growing inputs.",
        LOG_PREFIX, MAX_REF_IMAGES,
    )


