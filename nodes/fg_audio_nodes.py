"""
Audio nodes: Save Audio, the voice library, and everything the text-to-speech engines share
==========================================================================================
The engines themselves (fg_fish_audio, fg_higgs_audio) hold only their model code; their nodes are here.

Save Audio (V3 schema): saves audio with clean filenames and full control over the file.

Features:
- No counter suffix on the filename; %HMSf% timestamp by default, subfolders allowed
  ("dialogue/scene_01/%HMSf%"); collision suffix only when the name is taken
- Any folder: blank = ComfyUI's output folder, or an absolute path
- Overwrite switch for fixed names (a pipeline re-rendering "line_03" replaces it)
- The format picks its own settings (a DynamicCombo: only the chosen format's options show):
    flac  bit depth, compression level, sample rate, channels
    wav   bit depth (16 / 24 / 32 float), sample rate, channels
    mp3   quality (V0 or CBR), sample rate, channels
    opus  bitrate, application, sample rate, channels
- Optional peak normalisation, workflow metadata and a free-text comment tag
- Batches save as name_b0, name_b1, ...

┌─────────────────────────────────────┐
│ Save Audio                          │
├─────────────────────────────────────┤
│ ○  audio                            │
│ <→ TEXT filename_prefix> %HMSf%     │
│ <▼ format> flac / wav / mp3 / opus  │
│    └ that format's settings only    │
│ <▼ normalize>                       │
│ <BOOL overwrite> <BOOL metadata>    │
│ <→ TEXT output_folder> <→ comment>  │
└─────────────────────────────────────┘
"""
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field

import av
import numpy as np
import torch

import comfy.audio
import comfy.utils
import folder_paths
from aiohttp import web
from server import PromptServer
from comfy.cli_args import args
from comfy_api.latest import IO
from ._fg_helperfunctions import log, avoid_naming_collisions, get_output_path

SAVE_AUDIO_NODE_NAME = "Save Audio"

# wav codec + the sample format it is fed, per bit depth
WAV_CODECS = {
    "16": ("pcm_s16le", "s16"),
    "24": ("pcm_s24le", "s32"),
    "32 float": ("pcm_f32le", "flt"),
}
FLAC_BIT_DEPTHS = ["16", "24"]
MP3_QUALITIES = ["V0", "320k", "256k", "192k", "128k"]
OPUS_BITRATES = ["64k", "96k", "128k", "192k", "256k"]
OPUS_APPLICATIONS = ["audio", "voip", "lowdelay"]
PCM_RATES = ["original", "96000", "48000", "44100", "32000", "24000", "22050", "16000"]
MP3_RATES = ["original", "48000", "44100", "32000", "24000", "22050", "16000"]
OPUS_RATES = ["48000", "24000", "16000", "12000", "8000"]
CHANNELS = ["original", "mono", "stereo"]
NORMALIZE = ["off", "peak -0.1 dBFS", "peak -1 dBFS", "peak -3 dBFS", "peak -6 dBFS"]

# rates the lossy encoders accept; anything else moves to the next one up
_MP3_OK = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)
_OPUS_OK = (8000, 12000, 16000, 24000, 48000)


def _rate_input(options, default, tooltip):
    return IO.Combo.Input("sample_rate", options=options, default=default, tooltip=tooltip)


def _channels_input():
    return IO.Combo.Input("channels", options=CHANNELS, default="original",
                          tooltip="mono averages the channels; stereo duplicates mono. More than 2 keeps the first two.")


# ----------------------------------------------------------------------
# Audio shaping + writing (also used by the Fish S2 dialogue loop)
# ----------------------------------------------------------------------

def shape_audio(waveform: torch.Tensor, rate: int, sample_rate: str = "original", channels: str = "original",
                normalize: str = "off", fmt: str = "flac"):
    """[C, T] float -> converted [C, T] float32 and its rate."""
    wav = waveform.float()
    if channels == "mono" and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif channels == "stereo" and wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]  # the writers handle mono / stereo layouts
    target = rate if sample_rate in ("original", None) else int(sample_rate)
    allowed = {"opus": _OPUS_OK, "mp3": _MP3_OK}.get(fmt)
    if allowed and target not in allowed:
        target = next((r for r in allowed if r >= target), allowed[-1])
    if target != rate:
        wav = comfy.audio.resample(wav, rate, target)
    if normalize and normalize != "off":
        peak = float(wav.abs().max())
        if peak > 0:
            db = float(normalize.split()[1])
            wav = wav * (10 ** (db / 20) / peak)
    return wav.clamp(-1.0, 1.0), target


def write_audio(path: str, wav: torch.Tensor, rate: int, fmt: str, settings: dict | None = None,
                tags: dict | None = None):
    """Encode [C, T] float32 to path. settings = the chosen format's options (bit_depth, quality, ...)."""
    settings = settings or {}
    layout = "mono" if wav.shape[0] == 1 else "stereo"
    container = av.open(path, mode="w", format={"opus": "ogg"}.get(fmt, fmt))
    try:
        for key, value in (tags or {}).items():
            container.metadata[key] = value
        if fmt == "wav":
            codec, sample_fmt = WAV_CODECS[settings.get("bit_depth", "24")]
            stream = container.add_stream(codec, rate=rate, layout=layout)
            stream.format = sample_fmt
        elif fmt == "flac":
            stream = container.add_stream("flac", rate=rate, layout=layout)
            if settings.get("bit_depth", "24") == "16":
                stream.format = "s16"
            else:
                stream.format = "s32"
                stream.codec_context.bits_per_raw_sample = 24
            stream.codec_context.options = {"compression_level": str(int(settings.get("compression_level", 5)))}
        elif fmt == "mp3":
            stream = container.add_stream("libmp3lame", rate=rate, layout=layout)
            quality = settings.get("quality", "V0")
            if quality == "V0":
                stream.codec_context.qscale = 1
            else:
                stream.bit_rate = int(quality[:-1]) * 1000
        elif fmt == "opus":
            stream = container.add_stream("libopus", rate=rate, layout=layout)
            stream.bit_rate = int(settings.get("bitrate", "128k")[:-1]) * 1000
            stream.codec_context.options = {"application": settings.get("application", "audio")}
            for key, value in (tags or {}).items():  # Ogg keeps its tags on the stream
                stream.metadata[key] = value
        else:
            raise ValueError(f"{SAVE_AUDIO_NODE_NAME}: unknown format {fmt!r}")

        samples = np.ascontiguousarray(wav.movedim(0, 1).reshape(1, -1).numpy(), dtype=np.float32)
        frame = av.AudioFrame.from_ndarray(samples, format="flt", layout=layout)
        frame.sample_rate = rate
        frame.pts = 0
        container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    finally:
        container.close()


def _tags(embed_metadata, comment, prompt, extra_pnginfo) -> dict:
    tags = {}
    if comment and comment.strip():
        tags["comment"] = comment.strip()
    if embed_metadata and not args.disable_metadata:
        if prompt is not None:
            tags["prompt"] = json.dumps(prompt)
        for key, value in (extra_pnginfo or {}).items():
            tags[key] = json.dumps(value)
    return tags


# ----------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------

class FG_SaveAudio(IO.ComfyNode):
    """SaveAudio variant with clean filenames, any folder, and per-format encoding settings."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="FG_SaveAudio",
            display_name="🗑️ Save Audio",
            description="Saves audio with a clean filename, in any folder, as FLAC, WAV, MP3 or Opus.",
            category="Farrenzo's Garbage/Audio",
            search_aliases=["save audio", "export audio", "output audio", "write audio", "flac", "wav", "mp3", "opus"],
            inputs=[
                IO.Audio.Input("audio", tooltip="The audio to save."),
                IO.String.Input("filename_prefix", default="%HMSf%",
                    tooltip="File name without extension. Supports %HMSf% for a timestamp and subfolders: dialogue/scene_01/%HMSf%"),
                IO.DynamicCombo.Input(
                    "format",
                    tooltip="flac: lossless, smaller | wav: lossless, universal | mp3: lossy, plays everywhere | opus: lossy, smallest",
                    options=[
                        IO.DynamicCombo.Option("flac", [
                            IO.Combo.Input("bit_depth", options=FLAC_BIT_DEPTHS, default="24",
                                           tooltip="FLAC has no float; 24-bit keeps everything a float source holds that you can hear."),
                            IO.Int.Input("compression_level", default=5, min=0, max=12,
                                         tooltip="0 = fastest, 12 = smallest. Lossless at every level."),
                            _rate_input(PCM_RATES, "original", "Resample before saving."),
                            _channels_input(),
                        ]),
                        IO.DynamicCombo.Option("wav", [
                            IO.Combo.Input("bit_depth", options=list(WAV_CODECS), default="24",
                                           tooltip="32 float never clips; 16 is the most compatible."),
                            _rate_input(PCM_RATES, "original", "Resample before saving."),
                            _channels_input(),
                        ]),
                        IO.DynamicCombo.Option("mp3", [
                            IO.Combo.Input("quality", options=MP3_QUALITIES, default="V0",
                                           tooltip="V0 = best variable bitrate; the rest are constant bitrates."),
                            _rate_input(MP3_RATES, "original",
                                        "Resample before saving. Rates MP3 can't hold move to the next one up."),
                            _channels_input(),
                        ]),
                        IO.DynamicCombo.Option("opus", [
                            IO.Combo.Input("bitrate", options=OPUS_BITRATES, default="128k"),
                            IO.Combo.Input("application", options=OPUS_APPLICATIONS, default="audio",
                                           tooltip="audio: music / best quality | voip: tuned for speech | lowdelay: lowest latency"),
                            _rate_input(OPUS_RATES, "48000", "Opus only takes 8 / 12 / 16 / 24 / 48 kHz."),
                            _channels_input(),
                        ]),
                    ],
                ),
                IO.Combo.Input("normalize", options=NORMALIZE, default="off",
                               tooltip="Scale so the loudest sample sits at this level."),
                IO.Boolean.Input("overwrite", default=False,
                    tooltip="Replace a file of the same name instead of adding _01, _02 ... (for fixed names a pipeline re-renders)."),
                IO.Boolean.Input("embed_metadata", default=True, tooltip="Embed the ComfyUI workflow in the file's tags."),
                IO.String.Input("output_folder", default="", optional=True,
                    tooltip="Blank = ComfyUI's output folder. An absolute path saves there instead (no preview in the node)."),
                IO.String.Input("comment", default="", multiline=True, optional=True,
                    tooltip="Written to the file's comment tag, e.g. the line that was spoken."),
            ],
            outputs=[
                IO.Audio.Output("audio", tooltip="The audio, passed through."),
                IO.String.Output("filepath", tooltip="Full path of the saved file (one per line for a batch)."),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        audio,
        filename_prefix: str,
        format: dict,
        normalize: str = "off",
        overwrite: bool = False,
        embed_metadata: bool = True,
        output_folder: str = "",
        comment: str = ""
    ) -> IO.NodeOutput:
        if audio is None or audio.get("waveform") is None:
            raise ValueError(f"{SAVE_AUDIO_NODE_NAME}: no audio to save (the source may have no audio track).")
        fmt = format["format"]
        settings = {k: v for k, v in format.items() if k != "format"}

        output_dir = folder_paths.get_output_directory()
        output_folder = (output_folder or "").strip().strip('"')
        base = output_folder or output_dir
        if output_folder and not os.path.isabs(base):
            raise ValueError(f"{SAVE_AUDIO_NODE_NAME}: output_folder must be an absolute path (or blank for ComfyUI's output).")
        full_output_folder, filename_base, subfolder = get_output_path(
            node_name=SAVE_AUDIO_NODE_NAME, filename_prefix=filename_prefix, output_path=base)
        os.makedirs(full_output_folder, exist_ok=True)
        extension = f".{fmt}"
        hidden = cls.hidden
        tags = _tags(embed_metadata, comment, getattr(hidden, "prompt", None), getattr(hidden, "extra_pnginfo", None))
        in_output = os.path.normcase(os.path.abspath(base)) == os.path.normcase(os.path.abspath(output_dir))

        waveforms = audio["waveform"].detach().cpu()
        batch = waveforms.shape[0]
        results, paths = [], []
        for i in range(batch):
            name = filename_base if batch == 1 else f"{filename_base}_b{i}"
            filename = f"{name}{extension}" if overwrite else avoid_naming_collisions(full_output_folder, name, extension)
            filepath = os.path.join(full_output_folder, filename)
            wav, rate = shape_audio(waveforms[i], int(audio["sample_rate"]), settings.get("sample_rate", "original"),
                                    settings.get("channels", "original"), normalize, fmt)
            write_audio(filepath, wav, rate, fmt, settings, tags)
            paths.append(filepath)
            log(f"{SAVE_AUDIO_NODE_NAME}💾: Saved -> {filepath}", message_type="finish")
            if in_output:
                results.append({"filename": filename, "subfolder": subfolder, "type": "output"})

        return IO.NodeOutput(audio, "\n".join(paths), ui={"audio": results})


# ======================================================================
# Voice library: a folder of reference voices, scanned once and cached as JSON
# ======================================================================
#
# A voice is an audio file plus a transcript named after the audio file's full name + ".txt"
# (Audio_01.wav + Audio_01.wav.txt). Only that exact pair counts; everything else in the folder is
# ignored. Subfolders are scanned at any depth; a voice's id is its path relative to the folder
# ("Women/Audio_01.wav"). The scan is cached in ComfyUI/user/<user>/fg_voice_library/ and reused until
# the node's re-scan button is pressed. The voice folder itself is never written to.
#
# GET /fg/audio/voices?folder=<path>&rescan=0|1  ->  the cached (or fresh) scan, for the node's list.

VOICE_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}


def _voice_cache_dir() -> str:
    return os.path.join(folder_paths.get_user_directory(), "fg_voice_library")


def _voice_cache_path(folder: str) -> str:
    key = hashlib.sha1(os.path.normcase(os.path.abspath(folder)).encode("utf-8")).hexdigest()[:16]
    return os.path.join(_voice_cache_dir(), f"{key}.json")


def _read_transcript(path: str) -> str:
    try:
        with open(path, encoding="utf-8-sig") as f:
            return " ".join(f.read().split())
    except (OSError, UnicodeDecodeError):
        return ""


def _duration(path: str):
    try:
        import soundfile
        return round(soundfile.info(path).duration, 2)
    except Exception:  # noqa: BLE001 - informational only (m4a / some mp3 aren't readable by libsndfile)
        return None


def scan_voices(folder: str) -> dict:
    """Walk the folder, pair audio + transcript, write the cache, return it."""
    root = os.path.abspath(folder.strip().strip('"'))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Voice library: not a folder: {root}")
    voices = []
    for here, dirs, files in os.walk(root):
        dirs.sort()
        names = set(files)  # exact names as listed: "a.WAV.txt" does not pair with "a.wav"
        for f in sorted(files):
            if not f.endswith(".txt"):
                continue
            audio = f[:-4]
            if audio not in names or os.path.splitext(audio)[1].lower() not in VOICE_EXT:
                continue
            text = _read_transcript(os.path.join(here, f))
            if not text:
                continue
            path = os.path.join(here, audio)
            voices.append({"id": os.path.relpath(path, root).replace(os.sep, "/"), "file": audio, "transcript_file": f,
                           "transcript": text, "duration": _duration(path)})
    data = {"folder": root, "scanned": time.time(), "voices": voices}
    os.makedirs(_voice_cache_dir(), exist_ok=True)
    with open(_voice_cache_path(root), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    return data


def voice_library(folder: str, rescan: bool = False) -> dict:
    """The cached scan of a folder (scanned now if there is no cache, or on rescan)."""
    root = os.path.abspath(folder.strip().strip('"'))
    path = _voice_cache_path(root)
    if not rescan and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    return scan_voices(root)


def _voice_cache_stamp(folder: str) -> float:
    """When the folder was last scanned (so a re-scan re-runs the node)."""
    try:
        return os.path.getmtime(_voice_cache_path(os.path.abspath(folder.strip().strip('"'))))
    except OSError:
        return 0.0


def resolve_voice(lib: dict, ref: str) -> dict | None:
    """A voice by id ("Women/Audio_01.wav"), or by file name when that name is unique in the library."""
    ref = ref.strip().replace("\\", "/")
    by_id = {v["id"]: v for v in lib["voices"]}
    if ref in by_id:
        return by_id[ref]
    named = [v for v in lib["voices"] if v["file"] == os.path.basename(ref)]
    return named[0] if len(named) == 1 else None


@PromptServer.instance.routes.get("/fg/audio/voices")
async def _voices_route(request):
    folder = request.query.get("folder", "")
    if not folder.strip():
        return web.json_response({"folder": "", "voices": [], "error": "No folder set"})
    try:
        return web.json_response(voice_library(folder, rescan=request.query.get("rescan") == "1"))
    except FileNotFoundError as e:
        return web.json_response({"folder": folder, "voices": [], "error": str(e)})


# ======================================================================
# Text-to-speech: shared by both engines
# ======================================================================
#
#   TTS Loader --TTS_MODEL--> TTS Speak --AUDIO--> Save Audio
#   LoadAudio -> TTS Voice (speaker 0) -> TTS Voice (speaker 1) --TTS_VOICES--> TTS Speak
#   LoadAudio -> TTS Scene Turn -> TTS Scene Turn ... --TTS_HISTORY--> TTS Speak  (scene context)
#   Audio Library Loader --AUDIO_LIBRARY--> TTS Speak                             (voices from a folder)
#   JSON dialogue --> Audio Library Loader --> TTS Speak               (a whole dialogue, one file per line)
#
# Each engine is a package with the same three functions (fg_fish_audio, fg_higgs_audio):
#     detect(path) -> bool
#     load(path, opts) -> model
#     speak(model, text, speaker, voices, history, sampling, seed, opts, label) -> (AUDIO, info)
# The loader reads the engine from the checkpoint folder's contents; Speak calls whichever engine the
# model came from. What a line "hears" (history) is the engine's business: Fish hears every earlier
# line of the scene, Higgs only its own speaker's.
#
# Dialogue JSON (text, or a path to a .json file), one entry per line, in order:
#     {"dialogue_01.flac": {"audio_ref": "Audio_01.wav", "dialogue": "Hello there!"},
#      "dialogue_02.flac": {"audio_ref": "Audio_02.wav", "dialogue": "General Kenobi!"}}
# Each distinct audio_ref is a speaker, numbered in order of first appearance.

if "audios" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("audios", os.path.join(folder_paths.models_dir, "audios"), is_default=True)

TTS_ENGINES = {"fish": "Fish S2 Pro", "higgs": "Higgs v3"}
SAVE_FORMATS = ["flac", "wav", "mp3", "opus"]
_NO_CHECKPOINT = "(none found: use checkpoint_path)"
_DIALOGUE_FOLDER = "tts_dialogue"


def tts_engine(name: str):
    """The engine package (imported on first use: fish-speech's imports are heavy)."""
    if name == "fish":
        from . import fg_fish_audio as engine
    elif name == "higgs":
        from . import fg_higgs_audio as engine
    else:
        raise ValueError(f"No text-to-speech engine {name!r}")
    return engine


def detect_engine(path: str) -> str | None:
    """Which engine a checkpoint folder belongs to, from its contents (no heavy imports)."""
    if not os.path.isfile(os.path.join(path, "config.json")):
        return None
    if os.path.isfile(os.path.join(path, "codec.pth")):
        return "fish"
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            if "audio_encoder_config" in json.load(f):
                return "higgs"
    except (OSError, ValueError):
        pass
    return None


def tts_checkpoints() -> dict[str, list[str]]:
    """Checkpoint folders in the audios model folders (up to two levels down), by engine."""
    found: dict[str, set] = {e: set() for e in TTS_ENGINES}
    for base in folder_paths.get_folder_paths("audios"):
        if not os.path.isdir(base):
            continue
        for root, dirs, _files in os.walk(base):
            if os.path.relpath(root, base).count(os.sep) >= 1:
                dirs[:] = []
            engine = detect_engine(root) if root != base else None
            if engine:
                found[engine].add(os.path.relpath(root, base).replace(os.sep, "/"))
    return {e: sorted(v) for e, v in found.items()}


def resolve_checkpoint(checkpoint: str, checkpoint_path: str = "") -> str:
    if checkpoint_path.strip():
        return checkpoint_path.strip().strip('"')
    if not checkpoint or checkpoint == _NO_CHECKPOINT:
        raise ValueError("TTS Loader: set checkpoint_path, or put the checkpoint folder in ComfyUI/models/audios")
    for base in folder_paths.get_folder_paths("audios"):
        d = os.path.join(base, checkpoint)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"TTS Loader: {checkpoint} not found in the audios model folders")


def load_audio_file(path: str) -> dict:
    """An audio file as ComfyUI AUDIO."""
    from comfy_extras.nodes_audio import load
    waveform, sample_rate = load(path)
    return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}


def parse_dialogue(raw: str) -> list[dict]:
    """Dialogue JSON (text or a .json path) -> [{name, audio_ref, text}] in file order."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if not raw.startswith("{") and raw.lower().endswith(".json") and os.path.isfile(raw.strip('"')):
        with open(raw.strip('"'), encoding="utf-8-sig") as f:
            raw = f.read()
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise ValueError(f"Audio Library Loader: json_dialogue is not valid JSON ({e})") from e
    if not isinstance(data, dict):
        raise ValueError('Audio Library Loader: json_dialogue must be an object: {"line_01.flac": {"audio_ref": ..., "dialogue": ...}}')
    lines = []
    for name, entry in data.items():
        if not isinstance(entry, dict) or not str(entry.get("audio_ref", "")).strip() or not str(entry.get("dialogue", "")).strip():
            raise ValueError(f"Audio Library Loader: {name!r} needs both \"audio_ref\" and \"dialogue\"")
        lines.append({"name": str(name), "audio_ref": str(entry["audio_ref"]), "text": str(entry["dialogue"]).strip()})
    return lines


def line_path(folder: str, name: str, fmt: str) -> tuple[str, str]:
    """Where a dialogue line is saved, and in which format (the key's own extension wins)."""
    base, ext = os.path.splitext(name)
    if ext.lower().lstrip(".") in SAVE_FORMATS:
        fmt = ext.lower().lstrip(".")
    else:
        base = name
    return os.path.join(folder, f"{base}.{fmt}"), fmt


def status_line(info: dict) -> str:
    """One line of the Speak node's status (shown on the node by web/js/fg_tts_speak.js)."""
    name = f"{info['line']}: " if info.get("line") else ""
    if info.get("skipped"):
        return f"↺ {name}file exists, loaded"
    took = info["duration_s"] * info["rtf"]
    heard = f" · heard {info['heard']} earlier" if info.get("heard") else ""
    capped = " · ⚠ hit the length cap: probably rambled, retake" if info.get("capped") else ""
    return f"✓ {name}{info['duration_s']:.1f} s of speech in {took:.1f} s · RTF {info['rtf']} · {info.get('mode', '')}{heard}{capped}"


class LineProgress:
    """The node's progress bar and a console bar for one line.

    Generation stops at an end token, not at a known count, and the frame budget would leave the bar
    near 0%. So the total starts as an estimate from the text (~15 characters per second of speech at
    the engine's frame rate), grows if the line runs longer, and snaps to 100% when the line ends."""

    def __init__(self, text: str, budget: int, label: str = "TTS", frames_per_s: float = 21.5):
        from tqdm import tqdm

        words = re.sub(r"<\|[^|]*\|>|\[[^\]]*\]", " ", text)
        estimate = int(len(" ".join(words.split())) / 15 * frames_per_s) + 24
        self.total = max(16, min(int(budget), estimate))
        self.bar = comfy.utils.ProgressBar(self.total)
        self.console = tqdm(total=self.total, desc=label, unit="frame", leave=True, dynamic_ncols=True)
        self.current = 0

    def update(self, frames: int) -> None:
        if frames >= self.total * 0.95:
            self.total = int(frames * 1.2) + 8
            self.console.total = self.total
        self.bar.update_absolute(frames, self.total)
        self.console.update(frames - self.current)
        self.current = frames

    def done(self, frames: int) -> None:
        frames = max(frames, 1)
        self.total = frames
        self.console.total = frames
        self.console.update(frames - self.current)
        self.current = frames
        self.bar.update_absolute(frames, frames)
        self.console.close()


DECODE_BUCKET = 64  # frames


def decode_bucketed(decode, codes: torch.Tensor, dim: int, bucket: int = DECODE_BUCKET) -> torch.Tensor:
    """Run a codec's decode on codes padded to a multiple of `bucket` frames, and trim the audio back.

    On XPU every new input length builds new convolution kernels (~2 s per line); a handful of bucket
    lengths are built once and reused. The padding repeats the last frame; measured on Higgs it moves
    the kept audio by at most ~-36 dB, mostly in the last 50 ms. decode(codes) -> 1-D float samples."""
    n = codes.shape[dim]
    size = -(-n // bucket) * bucket
    if size > n:
        last = codes.narrow(dim, n - 1, 1)
        codes = torch.cat([codes, last.expand(*[size - n if d == dim % codes.dim() else -1
                                                for d in range(codes.dim())])], dim=dim)
    wav = decode(codes)
    return wav[: wav.numel() // size * n]


TAIL_DB = -60.0   # quieter than this is silence
TAIL_KEEP = 0.25  # seconds of silence kept after the last sound


def trim_tail(audio: dict, info: dict) -> tuple[dict, dict]:
    """Cut trailing silence off a spoken line. Higgs sometimes finishes the sentence and then emits silent
    frames until the length cap instead of its end marker (seen in the skill tests: ~7 s of -85 dB); such a
    take isn't a ramble, so it loses the `capped` flag. Returns the trimmed AUDIO and the updated info."""
    wav = audio["waveform"]
    rate = int(audio["sample_rate"])
    mono = wav[0].float().abs().amax(dim=0)  # [T] peak over channels
    win = max(1, rate // 50)                  # 20 ms
    n = mono.numel() // win
    if n < 2:
        return audio, info
    peaks = mono[: n * win].view(n, win).amax(dim=1)
    loud = (peaks > 10 ** (TAIL_DB / 20)).nonzero()
    if loud.numel() == 0:
        return audio, info
    end = min(mono.numel(), (int(loud[-1]) + 1) * win + int(TAIL_KEEP * rate))
    cut = (mono.numel() - end) / rate
    if cut < 0.5:
        return audio, info
    info = {**info, "trimmed_s": round(cut, 2), "duration_s": round(end / rate, 2)}
    if info.get("capped"):
        info["capped"] = False
        info["ended_in_silence"] = True
    return {"waveform": wav[..., :end], "sample_rate": rate}, info


@dataclass
class TTSModel:
    """What the loader hands to Speak: the engine, its loaded model, and the engine's own options."""
    engine: str
    model: object
    opts: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return TTS_ENGINES[self.engine]


TTSModelType = IO.Custom("TTS_MODEL")


# ----------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------

class FG_VoiceLibrary:
    """Load a folder of voice assets."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "tooltip": "Absolute path to the folder containing voice assets."}),
                # Managed by the node's checkbox list (web/js/fg_voice_library.js): a JSON list of voice ids.
                "selected_voices": ("STRING", {"default": "[]", "tooltip": "Ticked voices (set by the list below)."}),
            },
            "optional": {
                "json_dialogue": ("STRING", {
                    "forceInput": True,
                    "tooltip": "A dialogue to speak, one file per line: JSON text, or a path to a .json file."
                }),
            },
        }

    RETURN_TYPES = ("AUDIO_LIBRARY",)
    RETURN_NAMES = ("audio_selection",)
    FUNCTION = "load"
    CATEGORY = "Farrenzo's Garbage/Audio"
    DESCRIPTION = ("Voices from a folder (audio + its .txt transcript), scanned once and cached; tick the ones to use. "
                   "With a JSON dialogue, the voices it names are used and Speak renders every line.")

    @classmethod
    def IS_CHANGED(cls, folder_path, selected_voices, json_dialogue=None):
        return f"{folder_path}|{selected_voices}|{_voice_cache_stamp(folder_path) if folder_path.strip() else 0}"

    def load(self, folder_path, selected_voices, json_dialogue=None):
        if not folder_path.strip():
            raise ValueError("Audio Library Loader: set folder_path")
        lib = voice_library(folder_path)
        try:
            ticked = [str(v) for v in json.loads(selected_voices or "[]")]
        except ValueError:
            ticked = []
        lines = parse_dialogue(json_dialogue)
        if lines and any(resolve_voice(lib, ln["audio_ref"]) is None for ln in lines):
            lib = voice_library(folder_path, rescan=True)  # the folder may have changed since the last scan
        missing = sorted({ln["audio_ref"] for ln in lines if resolve_voice(lib, ln["audio_ref"]) is None})
        if missing:
            raise ValueError(f"Audio Library Loader: not in {lib['folder']} (or missing its .txt): {', '.join(missing)}")

        # speakers: a dialogue's voices in order of first appearance; otherwise the ticked voices in list order
        if lines:
            order = []
            for ln in lines:
                vid = resolve_voice(lib, ln["audio_ref"])["id"]
                ln["voice"] = vid
                if vid not in order:
                    order.append(vid)
        else:
            order = [v["id"] for v in lib["voices"] if v["id"] in ticked]
            gone = [t for t in ticked if t not in {v["id"] for v in lib["voices"]}]
            if gone:
                raise ValueError(f"Audio Library Loader: ticked voices no longer in the folder (re-scan): {', '.join(gone)}")
        by_id = {v["id"]: v for v in lib["voices"]}
        voices = [{"id": vid, "path": os.path.join(lib["folder"], vid), "text": by_id[vid]["transcript"]} for vid in order]
        for ln in lines:
            ln["speaker"] = order.index(ln["voice"])
        out = {"folder": lib["folder"], "voices": voices, "dialogue": lines or None}
        return {"ui": {"voices_used": order}, "result": (out,)}


class FG_TTSVoice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Reference voice, ideally 5-20 s of clean speech."}),
                "transcript": ("STRING", {"multiline": True, "default": "", "tooltip": "Exactly what the reference says."}),
            },
            "optional": {
                "voices": ("TTS_VOICES", {"tooltip": "Earlier voices; this one becomes the next speaker number."}),
            },
        }

    RETURN_TYPES = ("TTS_VOICES",)
    RETURN_NAMES = ("voices",)
    FUNCTION = "add"
    CATEGORY = "Farrenzo's Garbage/Audio"
    DESCRIPTION = "Add a reference voice. Chain them: the first is speaker 0, the next speaker 1, and so on."

    def add(self, audio, transcript, voices=None):
        if not transcript.strip():
            raise ValueError("TTS Voice: the transcript is required (exactly what the reference audio says)")
        return ((list(voices or []) + [{"audio": audio, "text": " ".join(transcript.split())}]),)


class FG_TTSSceneTurn:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The take chosen for this earlier line."}),
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "The text that take was spoken from."}),
                "speaker": ("INT", {"default": 0, "min": 0, "max": 15, "tooltip": "Who said it (voice order)."}),
            },
            "optional": {
                "history": ("TTS_HISTORY", {"tooltip": "The turns before this one."}),
            },
        }

    RETURN_TYPES = ("TTS_HISTORY",)
    RETURN_NAMES = ("history",)
    FUNCTION = "add"
    CATEGORY = "Farrenzo's Garbage/Audio"
    DESCRIPTION = ("An earlier line of the scene, so the next line is spoken hearing it. Chain them oldest first. "
                   "Fish hears every turn; Higgs only the turns of the speaker it is voicing.")

    def add(self, audio, text, speaker, history=None):
        return ((list(history or []) + [{"audio": audio, "text": text.strip(), "speaker": int(speaker)}]),)


class FG_TTSLoader(IO.ComfyNode):
    """One loader for both engines; the engine's own options show only for that engine."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        found = tts_checkpoints()

        def checkpoint_input(engine):
            opts = found[engine] or [_NO_CHECKPOINT]
            return IO.Combo.Input("checkpoint", options=opts, default=opts[0],
                                  tooltip="A checkpoint folder in models/audios (the list holds this engine's only).")

        def precision_input():
            return IO.Combo.Input("precision", options=["bfloat16", "float16"], default="bfloat16",
                                  tooltip="The language model; the codec has its own setting.")

        return IO.Schema(
            node_id="FG_TTSLoader",
            display_name="🗑️ TTS Loader",
            description="Load a text-to-speech model (Fish S2 Pro or Higgs v3). ComfyUI moves it to the GPU when a line is spoken.",
            category="Farrenzo's Garbage/Audio",
            search_aliases=["fish", "higgs", "tts", "text to speech", "voice clone"],
            inputs=[
                IO.DynamicCombo.Input("engine", tooltip="Fish S2 Pro: 44.1 kHz, hears the whole scene, [free text] tags. "
                                                        "Higgs v3: 24 kHz, hears its own speaker, 43 fixed <|tags|>.", options=[
                    IO.DynamicCombo.Option("fish", [
                        checkpoint_input("fish"),
                        IO.Int.Input("max_seq_len", default=8192, min=4096, max=32768, step=1024,
                                     tooltip="Context: voice references + earlier lines + this line. 8192 holds about 4 min of "
                                             "scene; the model's 32768 costs ~5.8 GB more (KV cache + mask)."),
                        precision_input(),
                        IO.Combo.Input("codec_precision", options=["float32", "bfloat16"], default="float32",
                                       tooltip="The codec is small; float32 avoids decode artefacts."),
                        IO.Combo.Input("mode", options=["graph", "nograph", "stock"], default="graph",
                                       tooltip="graph: faster loop + XPU graph | nograph: faster loop, eager | stock: fish-speech's own loop"),
                        IO.Int.Input("capture_step", default=4, min=1, max=9001,
                                     tooltip="Frames decoded eagerly (warm-up) before the XPU graph is captured. Graph mode only."),
                    ]),
                    IO.DynamicCombo.Option("higgs", [
                        checkpoint_input("higgs"),
                        precision_input(),
                        IO.Combo.Input("mode", options=["graph", "nograph", "eager"], default="graph",
                                       tooltip="graph: fixed-size cache + XPU graph | nograph: the same loop, eager | "
                                               "eager: transformers' own forward"),
                        IO.Int.Input("capture_step", default=4, min=1, max=9001,
                                     tooltip="Frames decoded eagerly (warm-up) before the XPU graph is captured. Graph mode only."),
                        IO.Int.Input("context_lines", default=1, min=0, max=4,
                                     tooltip="Scene context: how many of the speaker's own earlier lines go in the reference "
                                             "slot after the voice sample (Higgs never hears other speakers). 0 = plain cloning."),
                        IO.String.Input("codec_config", default="",
                                        tooltip="Path to a Higgs v2 codec config.json. Blank = one next to the checkpoint "
                                                "(higgs-audio-v2-3B/tokenizer/config.json), else the copy in fg_higgs_audio."),
                    ]),
                ]),
                IO.String.Input("checkpoint_path", default="", optional=True,
                                tooltip="Absolute path to a checkpoint folder; overrides the list. The engine is read from "
                                        "the folder's contents."),
            ],
            outputs=[TTSModelType.Output("model", tooltip="The model, for TTS Speak.")],
        )

    @classmethod
    def execute(cls, engine: dict, checkpoint_path: str = "") -> IO.NodeOutput:
        chosen = engine["engine"]
        opts = {k: v for k, v in engine.items() if k not in ("engine", "checkpoint")}
        path = resolve_checkpoint(engine.get("checkpoint", ""), checkpoint_path)
        sensed = detect_engine(path)
        if sensed is None:
            raise ValueError(f"TTS Loader: {path} isn't a Fish S2 Pro or Higgs v3 checkpoint")
        if sensed != chosen:
            log(f"TTS Loader: {path} is a {TTS_ENGINES[sensed]} checkpoint; loading it as one (with its default options)",
                message_type="warning")
            opts = {}
        model = tts_engine(sensed).load(path, opts)
        return IO.NodeOutput(TTSModel(sensed, model, opts))


class FG_TTSSpeak:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TTS_MODEL",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "One line. Fish: inline [tags] such as [whisper], [laughing], [pause]. Higgs: <|emotion:anger|>, "
                               "<|style:whispering|>, <|sfx:laughter|>Haha, ... Ignored when the voice library carries a JSON dialogue."
                }),
                "speaker": ("INT", {"default": 0, "min": 0, "max": 15, "tooltip": "Which voice speaks (voice order)."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True}),
                "temperature": ("FLOAT", {
                    "default": 0.8, "min": 0.05, "max": 1.9, "step": 0.05,
                    "tooltip": "Expressiveness. Higher values are more varied, lower values are more consistent. "
                               "Suggested: Fish 0.8, Higgs 0.65."
                }),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.01, "tooltip": "Diversity via nucleus sampling."}),
                "top_k": ("INT", {
                    "default": 30, "min": 1, "max": 1024,
                    "tooltip": "Stability. Only the k most likely tokens are considered at each step. Lower values are steadier, "
                               "higher values allow rarer, more surprising delivery."
                }),
                "max_new_tokens": ("INT", {
                    "default": 0, "min": 0, "max": 32768,
                    "tooltip": "Cap on audio frames (Fish ~21.5 per second, Higgs 25). 0 = the engine's default: Fish up to the "
                               "context limit; Higgs ~2.5x the line's estimated length + 3 s (it sometimes rambles past the end)."
                }),
            },
            "optional": {
                "voices": ("TTS_VOICES",),
                "history": ("TTS_HISTORY", {"tooltip": "Earlier lines of the scene (Fish hears them all, Higgs its speaker's)."}),
                "voice_library_input": ("AUDIO_LIBRARY", {"tooltip": "A folder of voices and their reference files."}),
                # dialogue mode only (the library carries a JSON dialogue)
                "save_folder": ("STRING", {"default": "",
                                "tooltip": f"Dialogue mode: where each line is saved as it finishes. Blank = ComfyUI output/{_DIALOGUE_FOLDER}."}),
                "save_format": (SAVE_FORMATS, {"default": "flac",
                                "tooltip": "Dialogue mode: format for line names without an extension (a name's own .flac / .wav / "
                                           ".mp3 / .opus wins). WAV and FLAC are 24-bit."}),
                "skip_existing": ("BOOLEAN", {"default": True,
                                  "tooltip": "Dialogue mode: a line whose file already exists is loaded instead of spoken again, "
                                             "so a run that stopped part-way picks up where it left off."}),
                "scene_context": ("BOOLEAN", {"default": True,
                                  "tooltip": "Dialogue mode: each line is spoken hearing the lines before it (Fish: all of them; "
                                             "Higgs: its own speaker's, up to the loader's context_lines)."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "info", "name")
    OUTPUT_IS_LIST = (True, True, True)
    OUTPUT_TOOLTIPS = ("One per line (a dialogue gives several; the nodes after this run once per line).",
                       "Timing per line (JSON).",
                       "File name per line (dialogue mode), or %HMSf% for a single line: wire into Save Audio's filename_prefix.")
    FUNCTION = "speak"
    OUTPUT_NODE = True  # dialogue mode saves its own files, so it runs with nothing wired after it
    CATEGORY = "Farrenzo's Garbage/Audio"
    DESCRIPTION = ("Speak one line with the loaded text-to-speech model (Fish S2 Pro or Higgs v3), as the next turn of a scene; "
                   "or, with a JSON dialogue from the voice library, every line in order, each saved as it finishes.")

    @classmethod
    def IS_CHANGED(cls, save_folder="", **kwargs):
        """Dialogue mode depends on what is already on disk (skip_existing): run again when the save
        folder's files change, e.g. a line was deleted to redo it."""
        folder = (save_folder or "").strip().strip('"') or os.path.join(folder_paths.get_output_directory(), _DIALOGUE_FOLDER)
        try:
            files = sorted((e.name, e.stat().st_size, e.stat().st_mtime_ns) for e in os.scandir(folder) if e.is_file())
        except OSError:
            files = []
        return hashlib.sha1(repr(files).encode()).hexdigest()

    def speak(self, model, text, speaker, seed, temperature, top_p, top_k, max_new_tokens, voices=None, history=None,
              voice_library_input=None, save_folder="", save_format="flac", skip_existing=True, scene_context=True):
        engine = tts_engine(model.engine)
        sampling = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_new_tokens": max_new_tokens}
        library_voices = [{"audio": load_audio_file(v["path"]), "text": v["text"]}
                          for v in (voice_library_input or {}).get("voices", [])]
        if (voice_library_input or {}).get("dialogue"):
            return self._dialogue(model, engine, voice_library_input["dialogue"], library_voices, seed, sampling,
                                  save_folder, save_format, skip_existing, scene_context)

        all_voices = list(voices or []) + library_voices
        if not text.strip():
            raise ValueError(f"{model.label} Speak: nothing to say")
        if all_voices and speaker >= len(all_voices):
            raise ValueError(f"{model.label} Speak: speaker {speaker} but only {len(all_voices)} voice(s) connected")
        audio, info = trim_tail(*engine.speak(model.model, text, speaker, all_voices, history, sampling, seed, model.opts,
                                              model.label))
        return {"ui": {"tts_status": [status_line(info)], "tts_lines": [info]},
                "result": ([audio], [json.dumps(info)], ["%HMSf%"])}

    def _dialogue(self, model, engine, lines, voices, seed, sampling, save_folder, save_format, skip_existing, scene_context):
        """Every line in order; each one saved the moment it is done."""
        import comfy.model_management as mm

        folder = save_folder.strip().strip('"') or os.path.join(folder_paths.get_output_directory(), _DIALOGUE_FOLDER)
        os.makedirs(folder, exist_ok=True)
        audios, infos, names, done = [], [], [], []
        for i, ln in enumerate(lines):
            mm.throw_exception_if_processing_interrupted()
            path, fmt = line_path(folder, ln["name"], save_format)
            if skip_existing and os.path.isfile(path):
                audio = load_audio_file(path)
                info = {"line": ln["name"], "skipped": "file exists", "file": path}
            else:
                audio, info = trim_tail(*engine.speak(model.model, ln["text"], ln["speaker"], voices,
                                                      done if scene_context else None, sampling, seed + i, model.opts,
                                                      f"{model.label} line {i + 1}/{len(lines)}"))
                wav, rate = shape_audio(audio["waveform"][0].cpu(), audio["sample_rate"], fmt=fmt)
                write_audio(path, wav, rate, fmt, tags={"comment": ln["text"]})
                info = {"line": ln["name"], "file": path, **info}
                log(f"{model.label} 💾: line {i + 1}/{len(lines)} saved -> {path}", message_type="finish")
            done.append({"audio": audio, "text": ln["text"], "speaker": ln["speaker"]})
            audios.append(audio)
            infos.append(info)
            names.append(os.path.splitext(os.path.basename(path))[0])
        spoken = sum(1 for i in infos if not i.get("skipped"))
        status = [f"{len(lines)} lines ({spoken} spoken, {len(lines) - spoken} loaded) saved to {folder}"]
        return {"ui": {"tts_status": status + [status_line(i) for i in infos], "tts_lines": infos},
                "result": (audios, [json.dumps(i) for i in infos], names)}


AUDIO_NODE_CLASS_MAPPINGS = {
    "FG_SaveAudio": FG_SaveAudio,
    "FG_VoiceLibrary": FG_VoiceLibrary,
    "FG_TTSVoice": FG_TTSVoice,
    "FG_TTSSceneTurn": FG_TTSSceneTurn,
    "FG_TTSLoader": FG_TTSLoader,
    "FG_TTSSpeak": FG_TTSSpeak,
}

AUDIO_NODE_DISPLAY_NAME_MAPPINGS = {
    "FG_SaveAudio": "🗑️ Save Audio",
    "FG_VoiceLibrary": "🗑️ Audio Library Loader",
    "FG_TTSVoice": "🗑️ TTS Voice (reference)",
    "FG_TTSSceneTurn": "🗑️ TTS Scene Turn (history)",
    "FG_TTSLoader": "🗑️ TTS Loader",
    "FG_TTSSpeak": "🗑️ TTS Speak",
}
