"""CLIP pass-through nodes that control the XPU graph patches at runtime."""

import importlib
import logging
import sys

from . import yue2_xpu_patch, minimax_music_xpu_patch

yue2_xpu_patch.install()
minimax_music_xpu_patch.install()


def _make_node(module_name, label, step_tooltip):
    class _Accel:
        @classmethod
        def INPUT_TYPES(cls):
            patch = sys.modules[module_name]
            return {
                "required": {
                    "clip": ("CLIP",),
                    "mode": (list(patch.MODES), {"default": "graph",
                             "tooltip": "graph: patched + XPU graph | nograph: patched, eager | stock: untouched ComfyUI"}),
                    "capture_step": ("INT", {"default": 4, "min": 1, "max": 9001, "tooltip": step_tooltip}),
                    "reload_patch": ("BOOLEAN", {"default": False,
                                     "tooltip": "Re-import the patch file before running, so edits apply without a restart."}),
                }
            }

        RETURN_TYPES = ("CLIP",)
        FUNCTION = "apply"
        CATEGORY = "FG/audio"
        DESCRIPTION = f"Route the {label} CLIP through this node to pick the XPU decode path."

        @classmethod
        def IS_CHANGED(cls, clip, mode, capture_step, reload_patch):
            return float("nan") if reload_patch else f"{mode}:{capture_step}"

        def apply(self, clip, mode, capture_step, reload_patch):
            patch = sys.modules[module_name]
            if reload_patch:
                patch = importlib.reload(patch)
                patch.install()
                logging.info("[FG] %s reloaded", module_name.rsplit(".", 1)[-1])
            patch.MODE = mode
            patch.CAPTURE_AT = capture_step
            return (clip,)

    return _Accel


FG_YuE2XPUAccel = _make_node(
    yue2_xpu_patch.__name__, "YuE2",
    "Tokens decoded eagerly (warm-up) before the XPU graph is captured. Only used in graph mode.")
FG_MiniMaxMusicXPUAccel = _make_node(
    minimax_music_xpu_patch.__name__, "MiniMax Music 3",
    "Frames decoded eagerly (warm-up) before the XPU graph is captured. Only used in graph mode.")