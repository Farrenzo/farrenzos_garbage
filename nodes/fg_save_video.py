"""
Save Video - ComfyUI Node
=================================
Saves videos with clean filenames and format options.

Features:
- No counter suffix on the filename; %HMSf% timestamp by default
- Collision handling for the rare microsecond clash
- Format choice: MP4 (via ComfyUI's native save path) or WebM (via PyAV)
- Optional metadata embedding
- WebM options: codec (vp9/av1), lossless, crf, av1 speed preset
- WebM path carries audio and vp9 alpha through when present

┌─────────────────────────────────────┐
│ Save Video                          │
├─────────────────────────────────────┤
│ ○  Videos                           │
│ <→ TEXT INPUT_File Name>            │
│ - Auto Populated to %HMSf%          │
│                                     │
│ <▼ FORMAT> Auto / .MP4 / .WebM      │
│ <BOOL _ embed_metadata>             │
│                                     │
│ ── WebM Options ──                  │
│ <▼ webm_codec>                      │
│ <BOOL _ webm_lossless>              │
│ <INT  _ webm_crf>                   │
│ <INT  _ webm_av1_preset>            │
└─────────────────────────────────────┘
"""
import os
import json


from fractions import Fraction

import av
import torch
import numpy as np

import folder_paths
from comfy.cli_args import args
from ._fg_helperfunctions import log, avoid_naming_collisions, get_output_path


class FG_SaveVideo:
    """
    SaveVideo variant with cleaner filenames and format options.

    MP4 is delegated to VideoInput.save_to(), which preserves audio and can
    stream-copy an already-compatible H.264 track instead of re-encoding.

    WebM is encoded manually, because comfy_api's VideoContainer enum only
    defines AUTO and MP4 -- save_to() has no WebM target at all.
    """

    FORMATS = ["auto", "mp4", "webm"]
    WEBM_CODECS = ["vp9", "av1"]

    # WebM container codec name -> ffmpeg encoder name
    _CODEC_MAP = {"vp9": "libvpx-vp9", "av1": "libsvtav1"}

    # Opus is fixed at 48 kHz; source audio is resampled to match.
    OPUS_RATE = 48000
    AUDIO_CHUNK = 1024

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.NODE_NAME = "Save Video"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "The video to save."}),
                "filename_prefix": ("STRING", {
                    "default": "%HMSf%",
                    "tooltip": "The prefix for the video to save. Supports %HMSf% for timestamp."
                }),
                "format": (s.FORMATS, {
                    "default": "auto",
                    "tooltip": "Container: mp4 (native save path, keeps audio) or webm (re-encoded via PyAV). 'auto' resolves to mp4."
                }),
                "embed_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Embed ComfyUI workflow metadata in the video container."
                }),
            },
            "optional": {
                # WebM-specific options (ignored for MP4)
                "webm_codec": (s.WEBM_CODECS, {
                    "default": "vp9",
                    "tooltip": "WebM only: vp9 supports alpha; av1 gives better compression but encodes slower."
                }),
                "webm_lossless": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "WebM only: vp9 lossless mode. Ignores crf and produces very large files. No effect on av1."
                }),
                "webm_crf": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 63,
                    "step": 1,
                    "tooltip": "WebM only: quality. Lower = better quality and bigger files. Ignored if lossless=True."
                }),
                "webm_av1_preset": ("INT", {
                    "default": 6,
                    "min": 0,
                    "max": 13,
                    "step": 1,
                    "tooltip": "WebM only, av1 only: encoder speed. 0=slowest/best, 13=fastest/worst."
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "Farrenzo's Garbage/Video"
    DESCRIPTION = "Saves a video with a clean timestamped filename, as MP4 or WebM."

    # ------------------------------------------------------------------
    # Path handling
    # ------------------------------------------------------------------

    def _build_metadata(self, embed_metadata, prompt, extra_pnginfo):
        """Collect workflow metadata, or None if disabled."""
        if not embed_metadata or args.disable_metadata:
            return None

        metadata = {}
        if extra_pnginfo is not None:
            metadata.update(extra_pnginfo)
        if prompt is not None:
            metadata["prompt"] = prompt

        return metadata or None

    # ------------------------------------------------------------------
    # Format writers
    # ------------------------------------------------------------------

    def _save_mp4(self, video, filepath: str, metadata):
        """
        Delegate to ComfyUI's native video save path.

        This keeps the audio track and lets the backend stream-copy a source
        that is already H.264 rather than re-encoding it.
        """
        from comfy_api.util import VideoContainer, VideoCodec

        video.save_to(
            filepath,
            format=VideoContainer.MP4,
            codec=VideoCodec.AUTO,
            metadata=metadata,
        )

    def _save_webm(
        self,
        video,
        filepath: str,
        metadata,
        codec: str,
        lossless: bool,
        crf: int,
        av1_preset: int,
    ):
        """
        Encode WebM manually via PyAV.

        VideoContainer has no WEBM member, so there is no native save path.
        This pulls the decoded components out of the VIDEO object and drives
        the encoder directly.

        Stream declaration order matters: every stream must be added before
        the first container.mux() call. The container header is written on
        that first mux, and a stream added afterwards never receives a
        time_base, so all of its packets fail to rebase.
        """
        components = video.get_components()
        images = components.images
        frame_rate = components.frame_rate
        audio = getattr(components, "audio", None)
        alpha = getattr(components, "alpha", None)

        # Decode the audio payload up front. If it is malformed we find out
        # now, while we can still choose not to declare an audio stream at all.
        audio_samples, audio_rate = self._prepare_audio(audio)

        container = av.open(filepath, mode="w", format="webm")

        try:
            if metadata:
                for key, value in metadata.items():
                    container.metadata[key] = json.dumps(value)

            # Alpha is only representable in vp9's yuva420p.
            has_alpha_channel = images.shape[-1] == 4
            save_alpha = codec == "vp9" and (has_alpha_channel or alpha is not None)

            stream = container.add_stream(self._CODEC_MAP[codec], rate=frame_rate)
            stream.width = int(images.shape[-2])
            stream.height = int(images.shape[-3])
            stream.pix_fmt = (
                "yuva420p" if save_alpha
                else ("yuv420p10le" if codec == "av1" else "yuv420p")
            )
            stream.bit_rate = 0

            options = {}
            if codec == "vp9" and lossless:
                options["lossless"] = "1"
            else:
                options["crf"] = str(crf)
            if codec == "av1":
                options["preset"] = str(av1_preset)
            stream.options = options

            # Declared here, encoded later, but it must exist before any mux.
            audio_stream = None
            if audio_samples is not None:
                audio_stream = container.add_stream("libopus", rate=self.OPUS_RATE)
                audio_stream.layout = (
                    "mono" if audio_samples.shape[0] == 1 else "stereo"
                )

            for index, frame in enumerate(images):
                if save_alpha:
                    if has_alpha_channel:
                        rgba = frame[..., :4]
                    else:
                        # Alpha supplied as a separate mask channel
                        mask = alpha[index].unsqueeze(-1)
                        rgba = torch.cat([frame[..., :3], mask], dim=-1)
                    array = torch.clamp(rgba * 255, min=0, max=255)
                    array = array.to(device=torch.device("cpu"), dtype=torch.uint8).numpy()
                    av_frame = av.VideoFrame.from_ndarray(array, format="rgba")
                else:
                    array = torch.clamp(frame[..., :3] * 255, min=0, max=255)
                    array = array.to(device=torch.device("cpu"), dtype=torch.uint8).numpy()
                    av_frame = av.VideoFrame.from_ndarray(array, format="rgb24")

                for packet in stream.encode(av_frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)

            if audio_stream is not None:
                self._encode_webm_audio(
                    container, audio_stream, audio_samples, audio_rate
                )

        finally:
            container.close()

    def _prepare_audio(self, audio):
        """
        Normalise a ComfyUI AUDIO payload to (channels, samples) float32.

        Returns (None, None) when there is no usable audio, so the caller can
        skip declaring the stream entirely.
        """
        if audio is None:
            return None, None

        try:
            waveform = audio["waveform"]
            sample_rate = int(audio["sample_rate"])

            # (B, C, N) -> (C, N), first batch item only
            samples = waveform[0].to(device=torch.device("cpu"), dtype=torch.float32)
            samples = samples.contiguous().numpy().astype(np.float32)

            if samples.ndim == 1:
                samples = samples[np.newaxis, :]
            if samples.shape[0] > 2:
                # Opus in WebM: downmix anything exotic to the first two channels
                samples = samples[:2]
            if samples.shape[1] == 0:
                return None, None

            return samples, sample_rate

        except Exception as error:
            log(
                f"{self.NODE_NAME}💾: Audio payload unusable, saving video only: {error}",
                message_type="warning",
            )
            return None, None

    def _encode_webm_audio(self, container, audio_stream, samples, sample_rate):
        """
        Encode (channels, samples) float32 audio into an already-declared
        Opus stream.

        Audio is fed in small chunks with explicit timestamps. One giant
        AudioFrame is both memory-hostile and unreliable through the
        resampler, and Opus is fixed at 48 kHz regardless of the source rate.
        """
        try:
            layout = "mono" if samples.shape[0] == 1 else "stereo"
            resampler = av.AudioResampler(
                format="s16", layout=layout, rate=self.OPUS_RATE
            )

            source_time_base = Fraction(1, sample_rate)
            pts = 0

            for start in range(0, samples.shape[1], self.AUDIO_CHUNK):
                chunk = np.ascontiguousarray(
                    samples[:, start:start + self.AUDIO_CHUNK]
                )
                frame = av.AudioFrame.from_ndarray(
                    chunk, format="fltp", layout=layout
                )
                frame.sample_rate = sample_rate
                frame.time_base = source_time_base
                frame.pts = pts
                pts += chunk.shape[1]

                for resampled in resampler.resample(frame):
                    for packet in audio_stream.encode(resampled):
                        container.mux(packet)

            # Flush the resampler, then the encoder.
            for resampled in resampler.resample(None):
                for packet in audio_stream.encode(resampled):
                    container.mux(packet)

            for packet in audio_stream.encode(None):
                container.mux(packet)

        except Exception as error:
            log(
                f"{self.NODE_NAME}💾: Audio track could not be encoded, "
                f"video is saved without it: {error}",
                message_type="warning",
            )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def save_video(
        self,
        video,
        filename_prefix,
        format,
        embed_metadata,
        webm_codec="vp9",
        webm_lossless=False,
        webm_crf=32,
        webm_av1_preset=6,
        prompt=None,
        extra_pnginfo=None,
    ):
        resolved_format = "mp4" if format == "auto" else format
        extension = f".{resolved_format}"

        full_output_folder, filename_base, subfolder = get_output_path(
            node_name       = self.NODE_NAME, 
            filename_prefix = filename_prefix, 
            output_path     = self.output_dir
        )
        filename = avoid_naming_collisions(full_output_folder, filename_base, extension)
        filepath = os.path.join(full_output_folder, filename)

        metadata = self._build_metadata(embed_metadata, prompt, extra_pnginfo)

        if resolved_format == "mp4":
            self._save_mp4(video, filepath, metadata)
        else:
            self._save_webm(
                video,
                filepath,
                metadata,
                codec=webm_codec,
                lossless=webm_lossless,
                crf=webm_crf,
                av1_preset=webm_av1_preset,
            )

        log(f"{self.NODE_NAME}💾: Saved -> {filepath}", message_type="finish")

        results = [{
            "filename": filename,
            "subfolder": subfolder,
            "type": self.type,
        }]

        return {
            "ui": {"images": results, "animated": (True,)},
            "result": (video,),
        }


