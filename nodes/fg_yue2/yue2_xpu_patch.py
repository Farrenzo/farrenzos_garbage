"""
fg_yue2_xpu - XPU graph capture for ComfyUI's YuE2 ABC/music decode loop.

Lives in farrenzos_garbage/nodes/fg_yue2/. Installed by fg_yue2_node.py; the
FG_YuE2XPUAccel node switches MODE at runtime and can hot-reload this file.

Why: on XPU the stock loop runs ~700+ tiny kernel launches per token at ~48 us each,
so the card idles most of the time. ComfyUI already has a graphable decode path
(FixedKV cache + device-side sampling), but it is gated on CUDA in two places:
  1. FixedKV is only used if comfy_kitchen.flash_attention_decode exists for the device
     (CUDA/HIP only) -> here we supply an SDPA version for XPU.
  2. Graph capture lives in comfy.model_prefetch and is hard-wired to torch.cuda
     -> here the whole decode step (sample + 28 layers + lm_head) is captured once
        as a torch.xpu.XPUGraph and replayed for every remaining token.

MODE = "stock"    stock ComfyUI loop (no patch at all)
MODE = "nograph"  patched loop (XPU attention + new sampler) but never captures a graph
MODE = "graph"    patched loop + graph capture (default; FG_YUE2_GRAPH env var sets the startup default)

Note: sampling uses inverse-CDF with pre-drawn uniforms instead of torch.multinomial
(RNG inside a graph is fragile), so a given seed will not reproduce the stock output.
"""

import logging
import os

import torch

import comfy.model_management
import comfy.model_prefetch
import comfy.utils
import comfy.text_encoders.yue2 as yue2
from comfy.text_encoders.llama import FixedKV, rope_matrix

from . import _xpu_ar_common as common

log = logging.getLogger("FG_YuE2_XPU")

MODES = ("graph", "nograph", "stock")
CAPTURE_AT = 4  # eager warm-up steps before capture (oneDNN/oneMKL init lazily on first use)
MODE = {"0": "stock", "nograph": "nograph"}.get(os.environ.get("FG_YUE2_GRAPH", "1"), "graph")
_is_xpu = common.is_xpu


# --- 2. Graph-captured generate loop ------------------------------------------------

if not hasattr(yue2, "_fg_orig_generate"):
    yue2._fg_orig_generate = yue2.YuE2TEModel._generate
_orig_generate = yue2._fg_orig_generate


def _generate(self, prefix, seed, max_tokens, phase, dtype, negative=None, cfg_scale=1.0,
              legacy_off=False, **sampling):
    stock_args = (self, prefix, seed, max_tokens, phase, dtype, negative, cfg_scale, legacy_off)
    if MODE == "stock" or not _is_xpu(self.execution_device) or not hasattr(torch.xpu, "XPUGraph"):
        return _orig_generate(*stock_args, **sampling)

    if max(len(prefix), len(negative or [])) + max_tokens > self.config.max_position_embeddings:
        raise ValueError("YuE2 prompt plus generation budget exceeds the model context; "
                         "reduce the token budget or prompt length.")

    device = self.execution_device
    prefixes = [prefix] if cfg_scale == 1.0 else [prefix, negative]
    prefix_length = max(map(len, prefixes))

    # Per-layer prefetch queues do stream syncs every call: illegal during capture and
    # pure overhead when the model is fully resident anyway.
    old_prefetch = self.model.prefetch_dynamic_vbars
    self.model.prefetch_dynamic_vbars = False
    with common.fixed_kv_scope():
        logits, cache, _ = self._prefill(prefixes, prefix_length + max_tokens, dtype)

    if not isinstance(cache[0], FixedKV):
        log.warning("FixedKV cache not active (fixed_kv disabled?); using stock loop.")
        self.model.prefetch_dynamic_vbars = old_prefetch
        del logits, cache
        return _orig_generate(*stock_args, **sampling)

    common.zero_stale(cache, device)

    sampling = dict(sampling)
    penalty_window = sampling.pop("penalty_window")
    min_tokens = sampling["min_tokens"]
    end = yue2.ABC_END if phase == "abc" else yue2.MUSIC_END
    batch = len(prefixes)

    positions = torch.tensor([[len(p)] for p in prefixes], device=device, dtype=torch.long)
    decode_buffers = (torch.empty((batch, 1, self.config.hidden_size), device=device, dtype=dtype),
                      rope_matrix(self.model.compute_freqs_cis(positions, device)))
    recent = torch.full((1, penalty_window), yue2.EOD, device=device, dtype=torch.long)
    tokens = torch.empty((max_tokens,), device=device, dtype=torch.long)
    step_index = torch.zeros((), device=device, dtype=torch.long)
    generator = torch.Generator(device=device).manual_seed(seed)
    uniforms = torch.rand((max_tokens,), device=device, generator=generator, dtype=torch.float32)
    host = {"step": 0}

    neg_inf = torch.tensor(float("-inf"), device=device, dtype=logits.dtype)

    def head():
        guided = logits.clone() if cfg_scale == 1.0 else logits[1:] + cfg_scale * (logits[:1] - logits[1:])
        # Stock code blocks the end token with a Python `if step < min_tokens`, which would get
        # frozen into the graph. Do it on the device instead so capture can start right away.
        blocked = step_index < min_tokens
        guided[..., end] = torch.where(blocked, neg_inf, guided[..., end])
        # step=min_tokens makes distribution() skip its own host-side check.
        scores = yue2.distribution(guided, recent, min_tokens, phase, legacy_off=legacy_off, **sampling)
        if sampling["temperature"] == 0:
            next_id = scores.argmax(-1, keepdim=True)
        else:
            cdf = scores.float().softmax(-1).cumsum(-1)
            u = uniforms.index_select(0, step_index.view(1)).view(1, 1) * cdf[..., -1:]
            next_id = torch.searchsorted(cdf, u).clamp_max_(cdf.shape[-1] - 1)
        tokens.index_copy_(0, step_index.view(1), next_id.view(1))
        recent.scatter_(1, (step_index % penalty_window).view(1, 1), next_id)
        step_index.add_(1)
        decode_buffers[0].copy_(self.model.embed_tokens(next_id.expand(batch, 1), out_dtype=dtype))

    def step_fn():
        head()
        out = self.model(None, embeds=decode_buffers[0], past_key_values=cache, dtype=dtype,
                         position_ids=positions, attention_mask=None, decode_buffers=decode_buffers)
        logits.copy_(self.model.lm_head(out[0][:, -1]))
        positions.add_(1)

    progress = comfy.utils.ProgressBar(max_tokens)
    stream = torch.xpu.Stream(device=device)
    stream.wait_stream(torch.xpu.current_stream(device))
    graph, capture_tried = None, False
    capture_at = max(1, int(CAPTURE_AT))
    if MODE == "nograph":
        capture_tried = True
        log.info("YuE2 %s: mode=nograph, running patched loop eagerly", phase)
    history = []
    nan_warned = False
    try:
        with torch.xpu.stream(stream):
            for step in comfy.utils.model_trange(max_tokens, desc="YuE2 ABC sampling" if phase == "abc"
                                                 else "YuE2 music sampling", unit="token"):
                comfy.model_management.throw_exception_if_processing_interrupted()
                host["step"] = step
                if graph is None and not capture_tried and step >= capture_at:
                    capture_tried = True
                    graph = common.capture(step_fn, stream, f"YuE2 {phase}")
                    if graph is not None:
                        log.info("YuE2 %s: XPU graph captured at step %d", phase, step)
                if graph is not None:
                    graph.replay()
                else:
                    step_fn()
                progress.update_absolute(step + 1)
                if (step + 1) % 16 == 0 or step + 1 == max_tokens:
                    if not nan_warned and not bool(torch.isfinite(logits).all()):
                        nan_warned = True
                        log.warning("YuE2 %s: non-finite logits at step %d - output will be garbage", phase, step)
                    for token in tokens[len(history):step + 1].tolist():
                        if token == end:
                            return history, False
                        history.append(token)
    finally:
        torch.xpu.current_stream(device).wait_stream(stream)
        self.model.prefetch_dynamic_vbars = old_prefetch
        comfy.model_prefetch.cleanup_prefetch_queues()
        del graph
    logging.warning("YuE2 %s reached its token budget before the end token.", phase)
    return history, True


def install():
    """Idempotent; re-run after importlib.reload() to swap in the fresh functions."""
    common.install_attention()
    yue2.YuE2TEModel._generate = _generate


def uninstall():
    yue2.YuE2TEModel._generate = _orig_generate