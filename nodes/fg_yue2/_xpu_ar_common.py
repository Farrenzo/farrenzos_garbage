"""
Shared pieces for graph-capturing ComfyUI's autoregressive decode loops on Intel XPU.

ComfyUI's fixed-size KV cache (FixedKV) is graph-friendly, but it only turns on when
comfy_kitchen has a flash-decode kernel for the device (CUDA/HIP). This supplies an
SDPA version for XPU, scoped with `fixed_kv_scope()` so only the models we patch use it.
Not hot-reloaded by the nodes (it holds the stashed originals); edits here need a restart.
"""

import contextlib
import logging
import threading

import torch
import torch.nn.functional as F

import comfy_kitchen

log = logging.getLogger("FG_XPU_AR")
_STATE = threading.local()

if not hasattr(comfy_kitchen, "_fg_orig_fad_available"):
    comfy_kitchen._fg_orig_fad_available = getattr(comfy_kitchen, "flash_attention_decode_is_available", None)
    comfy_kitchen._fg_orig_fad = getattr(comfy_kitchen, "flash_attention_decode", None)
_orig_available = comfy_kitchen._fg_orig_fad_available
_orig_decode = comfy_kitchen._fg_orig_fad


def is_xpu(device):
    return torch.device(device).type == "xpu"


def _available(device):
    if is_xpu(device):
        return bool(getattr(_STATE, "on", False))
    return bool(_orig_available is not None and _orig_available(device))


def _decode(q, k, v, kv_lengths):
    """q: [B,1,H,D]  k,v: [B,capacity,KVH,D]  kv_lengths: [B] valid slots incl. current."""
    if q.device.type != "xpu":
        return _orig_decode(q, k, v, kv_lengths)
    batch, _, heads, _ = q.shape
    capacity, kv_heads = k.shape[1], k.shape[2]
    # Fixed shapes + device-side length mask = graph-safe (no host sync, no reallocation).
    valid = torch.arange(capacity, device=q.device).view(1, capacity) < kv_lengths.view(batch, 1)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=valid.view(batch, 1, 1, capacity),
        enable_gqa=heads != kv_heads,
    )
    return out.transpose(1, 2).contiguous()


def install_attention():
    comfy_kitchen.flash_attention_decode_is_available = _available
    comfy_kitchen.flash_attention_decode = _decode


@contextlib.contextmanager
def fixed_kv_scope():
    """Wrap init_kv_cache / prefill so FixedKV gets chosen on XPU."""
    _STATE.on = True
    try:
        yield
    finally:
        _STATE.on = False


def zero_stale(cache, device):
    """FixedKV is torch.empty. CUDA's kernel only reads valid slots; SDPA reads the full
    capacity, and 0-weight * NaN from stale VRAM = NaN. Zero past each row's length."""
    first = cache[0]
    stale = (torch.arange(first.key.shape[1], device=device).view(1, -1)
             >= first.seqlen.view(-1, 1).long()).view(first.key.shape[0], -1, 1, 1)
    for kv in cache:
        kv.key.masked_fill_(stale, 0)
        kv.value.masked_fill_(stale, 0)


def capture(step_fn, stream, name):
    graph = torch.xpu.XPUGraph()
    torch.xpu.synchronize()
    try:
        with torch.xpu.stream(stream):
            graph.capture_begin()
            try:
                step_fn()
            finally:
                graph.capture_end()
        return graph
    except Exception as exc:  # anything host-side inside the step breaks capture
        log.warning("%s: XPU graph capture failed, continuing eager: %s", name, exc)
        torch.xpu.synchronize()
        return None