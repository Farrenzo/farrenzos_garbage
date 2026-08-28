from __future__ import annotations

import torch
from typing import Callable
from ._fg_helperfunctions import log
import comfy.model_management as model_management


import os
import time
import threading

import logging
_LOG = logging.getLogger("FG_KeepAlive")

_INTERVAL = float(os.environ.get("FG_KEEPALIVE_SECS", "30"))
_DISABLED = os.environ.get("FG_KEEPALIVE_OFF") == "1"
_MAX_CONSECUTIVE_FAILURES = 5

_thread = None
_stop = threading.Event()


LOG_PREFIX = "[XPU Global VRAM V2]"
MIB = 1024**2

# A tolerance is required because allocator statistics and global
# memory telemetry may not be updated at exactly the same time.
VALIDATION_TOLERANCE = 512 * MIB

_original_get_free_memory = model_management.get_free_memory

_last_backend: str | None = None
_shown_warnings: set[str] = set()


def _set_backend(name: str) -> None:
    """Log only when the active telemetry backend changes."""

    global _last_backend

    if _last_backend == name:
        return

    _last_backend = name
    log(f"{LOG_PREFIX} Active memory backend: {name}", "warning")


def _get_mem_get_info_function() -> Callable:
    """
    Find torch.xpu.mem_get_info across PyTorch API layouts.

    Depending on the PyTorch build, it may be exported directly
    through torch.xpu or through torch.xpu.memory.
    """

    direct_function = getattr(torch.xpu, "mem_get_info", None)

    if callable(direct_function):
        return direct_function

    memory_module = getattr(torch.xpu, "memory", None)
    memory_function = getattr(
        memory_module,
        "mem_get_info",
        None,
    )

    if callable(memory_function):
        return memory_function

    raise AttributeError(
        "torch.xpu.mem_get_info() is unavailable"
    )


def _get_allocator_memory(
    device: torch.device,
) -> tuple[int, int, int]:
    """
    Return:
        active memory,
        reserved memory,
        reusable reserved cache.
    """

    stats = torch.xpu.memory_stats(device)

    active = int(
        stats.get("active_bytes.all.current", 0)
    )
    reserved = int(
        stats.get("reserved_bytes.all.current", 0)
    )

    # Defensive protection against inconsistent transient statistics.
    if reserved < active:
        reserved = active

    reusable = reserved - active

    return active, reserved, reusable


def _validate_mem_get_info(
    *,
    global_free: int,
    global_total: int,
    active: int,
    allocatable_total: int,
) -> None:
    """Reject impossible or obviously stale driver values."""

    if global_total <= 0:
        raise RuntimeError(
            f"invalid total memory: {global_total}"
        )

    if global_free < 0:
        raise RuntimeError(
            f"negative free memory: {global_free}"
        )

    if global_free > global_total:
        raise RuntimeError(
            "free memory is larger than total memory: "
            f"{global_free} > {global_total}"
        )

    # Physical total and allocatable total may differ slightly,
    # but a very large mismatch indicates incompatible telemetry.
    minimum_expected_total = int(
        allocatable_total * 0.75
    )
    maximum_expected_total = int(
        allocatable_total * 1.25
    )

    if not (
        minimum_expected_total
        <= global_total
        <= maximum_expected_total
    ):
        raise RuntimeError(
            "global total differs too much from allocatable total: "
            f"{global_total} vs {allocatable_total}"
        )

    global_used = global_total - global_free

    # Active PyTorch tensors must be represented inside globally
    # occupied device memory. If they are not, mem_get_info is
    # likely returning stale or incorrect values.
    if active > global_used + VALIDATION_TOLERANCE:
        raise RuntimeError(
            "global telemetry does not include current PyTorch "
            "allocations: "
            f"active={active}, global_used={global_used}"
        )


def _calculate_with_mem_get_info(
    device: torch.device,
    active: int,
    reusable: int,
    allocatable_total: int,
) -> int:
    """
    Preferred path.

    This mirrors the CUDA calculation used by ComfyUI:
        global free + reusable PyTorch cache.
    """

    mem_get_info = _get_mem_get_info_function()

    global_free, global_total = mem_get_info(device)

    global_free = int(global_free)
    global_total = int(global_total)

    _validate_mem_get_info(
        global_free=global_free,
        global_total=global_total,
        active=active,
        allocatable_total=allocatable_total,
    )

    available = global_free + reusable

    # ComfyUI should never be told that more memory is usable than
    # PyTorch reports as allocatable for this device.
    available = max(
        0,
        min(allocatable_total, available),
    )

    _set_backend("torch.xpu.mem_get_info")

    return available


def _patched_get_free_memory(
    dev=None,
    torch_free_too: bool = False,
):
    """
    Use device-wide XPU memory telemetry for Intel GPUs.

    Other device types continue using the original ComfyUI function.
    """

    if dev is None:
        dev = model_management.get_torch_device()

    if not model_management.is_device_xpu(dev):
        return _original_get_free_memory(
            dev,
            torch_free_too,
        )

    try:
        active, reserved, reusable = (
            _get_allocator_memory(dev)
        )

        allocatable_total = int(
            torch.xpu.get_device_properties(
                dev
            ).total_memory
        )

    except Exception as error:
        log(
            (
                f"{LOG_PREFIX} Allocator_stats_failed. | "
                "Cannot read XPU allocator statistics. | "
                f"Using original ComfyUI calculation.\n\n{error}"
            ),
            "error"
        )
        _set_backend("original ComfyUI fallback")
        return _original_get_free_memory(
            dev,
            torch_free_too,
        )

    try:
        available = _calculate_with_mem_get_info(
            device=dev,
            active=active,
            reusable=reusable,
            allocatable_total=allocatable_total,
        )

    except Exception as mem_get_info_error:
        log(
            (
                f"{LOG_PREFIX} mem_get_info_failed mem_get_info is unavailable"
                f"or invalid Using original ComfyUI calculation.\n{mem_get_info_error}"
            ),
            "error"
        )
        _set_backend("original ComfyUI fallback")
        return _original_get_free_memory(
            dev,
            torch_free_too,
        )

    if torch_free_too:
        return available, reusable

    return available


def install_xpu_patch() -> None:
    """Install the patch once during ComfyUI startup."""

    current_function = model_management.get_free_memory

    if getattr(
        current_function,
        "_xpu_global_vram_v2",
        False,
    ):
        log(f"{LOG_PREFIX} Patch is already installed.", "info")
        return

    if (
        not hasattr(torch, "xpu")
        or not torch.xpu.is_available()
    ):
        log(f"{LOG_PREFIX} Intel XPU is unavailable.", "error")
        return

    device = model_management.get_torch_device()

    if not model_management.is_device_xpu(device):
        log(f"{LOG_PREFIX} Current ComfyUI device is not XPU: {device}", "error")
        return

    setattr(
        _patched_get_free_memory,
        "_xpu_global_vram_v2",
        True,
    )

    model_management.get_free_memory = (
        _patched_get_free_memory
    )

    try:
        available, reusable = (
            _patched_get_free_memory(
                device,
                torch_free_too=True,
            )
        )

        allocatable_total = int(
            torch.xpu.get_device_properties(
                device
            ).total_memory
        )
        log(
            (
                f"{LOG_PREFIX} Patch installed."
                f"\nCurrent usable free: {available / 1024**3} GiB;"
                f"\nreusable PyTorch cache: {reusable / 1024**3} GiB;"
                f"\nallocatable total: {allocatable_total / 1024**3} GiB."
            ),
            "info"
        )
    except Exception:
        # The patched function already has safe fallbacks.
        log(f"{LOG_PREFIX} Initial diagnostic query failed.", "error")


# ----------------------------------- #
#  Keep the model in the GPU damnit!  #
# ----------------------------------- #

"""
FG_KeepAlive -- prevents Windows/WDDM from idle-evicting XPU allocations.

Windows trims a compute adapter's allocations after ~72s of inactivity. On a
headless Intel Arc this dumps everything resident into system RAM, and the
restore (~6s for 20GB over PCIe) blocks long enough to trip TDR, surfacing as
UR_RESULT_ERROR_DEVICE_LOST. Worse, the system-memory copy is never released,
so RAM climbs until eviction starts targeting the pagefile and stops finishing.

A trivial op submitted every N seconds resets the idle timer for the whole
process context, keeping every allocation resident. Cost is microseconds.

INSTALL
    Drop this file in your node pack, then in __init__.py:

        from .fg_keepalive import start_keepalive
        start_keepalive()

ENV
    FG_KEEPALIVE_SECS   interval in seconds (default 30; must stay under ~72)
    FG_KEEPALIVE_OFF    set to 1 to disable without editing code
"""

def _loop(interval):
    scratch = None
    pokes = 0
    failures = 0

    while not _stop.is_set():
        try:
            if scratch is None:
                # Allocated once and reused -- repeated tiny allocations would
                # churn the caching allocator for no reason.
                scratch = torch.ones(64, 64, device="xpu")
            scratch.mul_(1.0)          # write, so the page can't be considered clean
            float(scratch.sum())       # read + implicit sync
            pokes += 1
            failures = 0
            if pokes == 1:
                _LOG.info("FG_KeepAlive active (every %.0fs)", interval)
        except Exception as exc:
            failures += 1
            scratch = None             # force reallocation on the next attempt
            _LOG.warning("FG_KeepAlive poke failed (%d/%d): %r",
                         failures, _MAX_CONSECUTIVE_FAILURES, exc)
            if failures >= _MAX_CONSECUTIVE_FAILURES:
                _LOG.error("FG_KeepAlive giving up after %d consecutive failures",
                           failures)
                return
        _stop.wait(interval)


def start_keepalive(interval=None):
    """Idempotent. Safe to call on non-XPU systems -- it just does nothing."""
    global _thread

    if _DISABLED:
        _LOG.info("FG_KeepAlive disabled via FG_KEEPALIVE_OFF")
        return False

    if _thread is not None and _thread.is_alive():
        return True

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        _LOG.debug("FG_KeepAlive: no XPU device, not starting")
        return False

    interval = float(interval) if interval else _INTERVAL
    if interval >= 72:
        _LOG.warning("FG_KeepAlive interval %.0fs is at or past the observed "
                     "72s eviction threshold; clamping to 30s", interval)
        interval = 30.0

    _stop.clear()
    _thread = threading.Thread(
        target=_loop, args=(interval,), name="FG_KeepAlive", daemon=True
    )
    _thread.start()
    return True


def stop_keepalive():
    """Mainly useful for A/B testing whether it's still needed."""
    _stop.set()
    if _thread is not None:
        _thread.join(timeout=5)

