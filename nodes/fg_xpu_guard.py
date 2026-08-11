"""
FG_XPUGuard — Intel Arc / XPU device health gate for ComfyUI.

Drop into your node pack. Wire it as the first node in a workflow (passthrough
accepts any type) so the probe runs before anything expensive is scheduled.

What it does NOT do: recover a lost Level Zero context in-process. That is not
possible. Once the context dies, every allocation, queue and compiled module
tied to it is invalid, and torch.xpu offers no teardown/re-init path. This node
detects the condition, converts it into a clean failure instead of a mid-graph
traceback, and can optionally restart the process and re-queue the prompt.
"""

import json
import os
import sys
import threading
import time

import torch

RESUME_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fg_xpu_resume.json")

# Level Zero / UR errors that all mean "the context is gone, restart required".
FATAL_MARKERS = (
    "UR_RESULT_ERROR_DEVICE_LOST",
    "UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY",
    "UR_RESULT_ERROR_OUT_OF_RESOURCES",
    "UR_RESULT_ERROR_UNKNOWN",
    "Native API failed",
    "level_zero backend failed",
)


class AnyType(str):
    """Wildcard socket type — matches anything on the frontend."""

    def __ne__(self, other):
        return False


ANY = AnyType("*")


def is_fatal_backend_error(exc) -> bool:
    text = f"{exc}"
    return any(marker in text for marker in FATAL_MARKERS)


def vram_info(index: int = 0):
    """(free_mb, total_mb) or (None, None) if unavailable."""
    try:
        free, total = torch.xpu.mem_get_info(index)
        return free // (1024 * 1024), total // (1024 * 1024)
    except Exception:
        return None, None


def probe_xpu(index: int = 0, dim: int = 64):
    """
    Cheap full round trip: allocate -> compute -> synchronize -> read back.

    The readback matters. Some failure modes only surface on the device-to-host
    memcpy path, so a probe that stops at synchronize() will report healthy on a
    device that cannot actually return results.

    Returns (ok: bool, detail: str).
    """
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        return False, "torch.xpu reports unavailable"

    try:
        dev = torch.device(f"xpu:{index}")
        a = torch.randn(dim, dim, device=dev, dtype=torch.float32)
        b = a @ a.T
        torch.xpu.synchronize(dev)
        val = float(b.flatten()[0].item())
        del a, b
        if val != val:  # NaN
            return False, "probe returned NaN — device is computing garbage"
        return True, "ok"
    except Exception as exc:
        kind = "DEVICE LOST" if is_fatal_backend_error(exc) else "error"
        return False, f"{kind}: {type(exc).__name__}: {exc}"


def _restart_comfy(delay: float = 1.5):
    """Restart out-of-band so the current response can still be returned."""

    def _go():
        time.sleep(delay)
        # Preferred: ComfyUI-Manager's reboot endpoint, which restarts cleanly.
        try:
            import requests
            from server import PromptServer

            port = PromptServer.instance.port
            requests.get(f"http://127.0.0.1:{port}/api/manager/reboot", timeout=3)
            return
        except Exception:
            pass
        # Fallback: replace the process image. On Windows this is emulated —
        # the parent exits and a new process spawns, so console handles and any
        # supervising batch file may behave differently than on Linux.
        os.execv(sys.executable, [sys.executable] + sys.argv)

    threading.Thread(target=_go, daemon=True).start()


def _write_resume(prompt):
    """Persist the graph so it can be replayed after the restart."""
    try:
        with open(RESUME_FILE, "w", encoding="utf-8") as fh:
            json.dump({"saved_at": time.time(), "prompt": prompt}, fh)
    except Exception as exc:
        print(f"[FG_XPUGuard] could not write resume file: {exc}")


class FG_XPUGuard:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "on_device_lost": (
                    ["raise", "restart", "restart_and_resume"],
                    {"default": "raise"},
                ),
                "device_index": ("INT", {"default": 0, "min": 0, "max": 7}),
                "probe_dim": ("INT", {"default": 64, "min": 8, "max": 2048}),
                "min_free_vram_mb": (
                    "INT",
                    {"default": 0, "min": 0, "max": 131072, "step": 256},
                ),
            },
            "optional": {"passthrough": (ANY,)},
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = (ANY, "STRING")
    RETURN_NAMES = ("passthrough", "status")
    FUNCTION = "run"
    CATEGORY = "Farrenzo's Garbage/system"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # never cache — the whole point is to re-probe

    def run(
        self,
        on_device_lost,
        device_index,
        probe_dim,
        min_free_vram_mb,
        passthrough=None,
        prompt=None,
    ):
        ok, detail = probe_xpu(device_index, probe_dim)
        free_mb, total_mb = vram_info(device_index)
        vram_str = f"{free_mb}/{total_mb} MB free" if free_mb is not None else "VRAM unknown"

        if ok:
            # Preflight headroom check. Because XPU surfaces exhaustion as a
            # fatal backend error rather than a catchable OOM, aborting early is
            # far cheaper than eating a device reset halfway through sampling.
            if min_free_vram_mb and free_mb is not None and free_mb < min_free_vram_mb:
                raise RuntimeError(
                    f"[FG_XPUGuard] Only {free_mb} MB free on xpu:{device_index}, "
                    f"below the {min_free_vram_mb} MB floor. Aborting before the "
                    f"allocator can trigger a device reset."
                )
            status = f"healthy — xpu:{device_index}, {vram_str}"
            print(f"[FG_XPUGuard] {status}")
            return (passthrough, status)

        status = f"UNHEALTHY — xpu:{device_index}, {vram_str} — {detail}"
        print(f"[FG_XPUGuard] {status}")

        if on_device_lost == "raise":
            raise RuntimeError(
                f"[FG_XPUGuard] {detail}\n"
                f"The Level Zero context is gone. Loaded models, VAE and LoRA "
                f"weights on this device are unrecoverable and ComfyUI must be "
                f"restarted to obtain a fresh context."
            )

        if on_device_lost == "restart_and_resume" and prompt is not None:
            _write_resume(prompt)

        print("[FG_XPUGuard] restarting ComfyUI to rebuild the Level Zero context...")
        _restart_comfy()
        raise RuntimeError(f"[FG_XPUGuard] {detail} — restart in progress.")


def _resume_pending_prompt():
    """
    On startup, replay a graph saved by a previous restart_and_resume trigger.
    Runs once, then deletes the file so a crash loop can't spin forever.
    """
    if not os.path.exists(RESUME_FILE):
        return

    def _go():
        try:
            with open(RESUME_FILE, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            payload = None
        try:
            os.remove(RESUME_FILE)
        except Exception:
            pass
        if not payload:
            return

        # Wait for the server to bind and for the device to settle after reset.
        time.sleep(10)
        try:
            import requests
            from server import PromptServer

            port = PromptServer.instance.port
            requests.post(
                f"http://127.0.0.1:{port}/prompt",
                json={"prompt": payload["prompt"], "client_id": "fg_xpu_guard"},
                timeout=10,
            )
            print("[FG_XPUGuard] re-queued the prompt that was interrupted by device loss.")
        except Exception as exc:
            print(f"[FG_XPUGuard] resume failed: {exc}")

    threading.Thread(target=_go, daemon=True).start()


_resume_pending_prompt()
