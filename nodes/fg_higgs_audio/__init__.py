"""Higgs Audio v3 TTS 4B engine (bosonai/higgs-audio-v3-tts-4b). The nodes are in fg_audio_nodes.py (TTS
Loader / Speak); this package holds the model code (higgs_core.py).

Higgs has no multi-speaker conversation: one reference slot, then the line. So a line hears its speaker's
voice sample plus that speaker's own last `context_lines` lines (AUDIO_SPEC §6.7.2), never the other
speakers. Tags are fixed: <|emotion:*|>, <|style:*|>, <|prosody:*|>, <|sfx:*|> (see the checkpoint's
PROMPTING.md); anything else degrades the line or is read aloud.

License: the Higgs Audio v3 weights are under Boson AI's license (see the checkpoint's LICENSE).
"""
import json
import os
import re

from . import higgs_core as core

DEFAULT_MAX_FRAMES = 1500  # 25 frames per second: 60 s


def auto_cap(text: str) -> int:
    """Frames for max_new_tokens 0: ~2.5x the line's estimated length (15 characters per second) + 3 s.
    Higgs sometimes rambles past the end of a line (more often with context); this stops it early."""
    words = " ".join(re.sub(r"<\|[^|]*\|>", " ", text).split())
    seconds = max(1.5, len(words) / 15)
    return min(DEFAULT_MAX_FRAMES, int(seconds * core.FRAMES_PER_S * 2.5) + 3 * core.FRAMES_PER_S)


def detect(path: str) -> bool:
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            return "audio_encoder_config" in json.load(f)
    except (OSError, ValueError):
        return False


def load(path: str, opts: dict):
    return core.load(path, str(opts.get("codec_config") or "").strip().strip('"'), opts.get("precision", "bfloat16"))


def speak(model, text: str, speaker: int, voices: list | None, history: list | None, sampling: dict, seed: int,
          opts: dict, label: str) -> tuple[dict, dict]:
    """One line: the speaker's voice sample, then its own last `context_lines` lines from `history`."""
    voices = list(voices or [])
    refs = [voices[speaker]] if voices else []
    keep = int(opts.get("context_lines", 1))
    if refs and keep:
        refs += [h for h in (history or []) if h["speaker"] == speaker][-keep:]
    bad = core.unknown_tags(text)
    if bad:
        from .._fg_helperfunctions import log
        log(f"Higgs: tags it doesn't know (may be read aloud): {', '.join(bad)}", message_type="warning")
    audio, info = core.speak(model, text, refs, sampling["temperature"], sampling["top_p"], sampling["top_k"],
                             sampling["max_new_tokens"] or auto_cap(text), seed, label, opts.get("mode", "graph"),
                             int(opts.get("capture_step", 4)))
    info["heard"] = max(0, len(refs) - 1)
    return audio, info
