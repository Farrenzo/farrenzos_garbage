"""Fish Audio S2 Pro engine (fishaudio/s2-pro). The nodes are in fg_audio_nodes.py (TTS Loader / Speak);
this package holds the model code: fish_s2_core.py (loader, conversation, one line), fish_s2_fast.py
(the faster decode loop with XPU graph capture), fish_speech/ (never pip-installed) and descript_shim.py.

License: fish-speech and the S2 Pro weights are under the Fish Audio Research License
(fish_speech/LICENSE): research and non-commercial use.
"""
import os

from . import fish_s2_core as core


def detect(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json")) and os.path.isfile(os.path.join(path, "codec.pth"))


def load(path: str, opts: dict):
    return core.load(path, int(opts.get("max_seq_len", 8192)), opts.get("precision", "bfloat16"),
                     opts.get("codec_precision", "float32"))


def speak(model, text: str, speaker: int, voices: list | None, history: list | None, sampling: dict, seed: int,
          opts: dict, label: str) -> tuple[dict, dict]:
    """One line as the next turn of the scene: every earlier line in `history` is heard (oldest dropped when
    the context is full)."""
    audio, info = core.speak(model, text, speaker, voices, history, sampling["temperature"], sampling["top_p"],
                             sampling["top_k"], sampling["max_new_tokens"], seed, opts.get("mode", "graph"),
                             int(opts.get("capture_step", 4)), label=label)
    info["heard"] = info.get("history_used", 0)
    return audio, info
