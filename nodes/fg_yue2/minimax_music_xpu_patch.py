"""
XPU graph capture for ComfyUI's MiniMax Music 3 AR loop ("AR sampling").

One frame (25 per second of audio) on the stock loop is:
  c0 sample from the main LM (batch 2 for CFG) -> 7-ish sequential passes of the RVQ depth
  decoder, each with its own top-k sample -> embed the frame -> main LM forward,
plus a .item() host sync every frame. On XPU none of it is graphed, so a frame is
thousands of ~48 us launches (~290 ms/frame = 3.45 it/s).

This patch records one whole frame as a torch.xpu.XPUGraph and replays it:
  - FixedKV + SDPA decode attention on XPU (shared with the YuE2 patch)
  - top-k sampling via inverse-CDF on pre-drawn uniforms (graph-safe; seeds won't match stock)
  - explicit device-side RoPE positions (stock relies on a Python counter that a graph would freeze)
  - frame hiddens/codes written into preallocated buffers; stop token checked every
    CHECK_EVERY frames instead of every frame
MODE: "graph" | "nograph" | "stock" | "profile" (FG_MINIMAX_GRAPH env var sets the startup default).
"profile" runs CAPTURE_AT warm-up frames, captures each part of the frame (c0 sample, depth
decoder, embed, main LM) as its own graph, times PROFILE_FRAMES replays of each, logs the
breakdown plus model shapes/weight sizes, and stops (~1 s of audio; ignore it).
"""

import logging
import os
import threading
import time
from collections import defaultdict

import torch

import comfy.model_management
import comfy.model_prefetch
import comfy.utils
import comfy.ldm.minimax_music.ar as ar
from comfy.ldm.minimax_music.prompt import AUDIO_CODE_OFFSET, SPECIAL_TOKEN_IDS
from comfy.text_encoders.llama import FixedKV

from . import _xpu_ar_common as common

log = logging.getLogger("FG_MiniMaxMusic_XPU")

MODES = ("graph", "nograph", "stock", "profile")
PROFILE_FRAMES = 24
MODE = {"0": "stock", "nograph": "nograph"}.get(os.environ.get("FG_MINIMAX_GRAPH", "1"), "graph")
CAPTURE_AT = 4
CHECK_EVERY = 16

if not hasattr(ar, "_fg_orig_generate"):
    ar._fg_orig_generate = ar.MiniMaxMusic3AR.generate
    ar._fg_orig_sample_topk = ar.sample_topk
_orig_generate = ar._fg_orig_generate
_orig_sample_topk = ar._fg_orig_sample_topk

_SAMPLER = threading.local()  # .state = (uniforms, cursor) while our loop runs


def _sample_topk(logits, top_k, generator):
    state = getattr(_SAMPLER, "state", None)
    if state is None:
        return _orig_sample_topk(logits, top_k, generator)
    uniforms, cursor = state
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    top_k = min(top_k, values.shape[-1])
    threshold = torch.topk(values, top_k, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    cdf = torch.softmax(values, dim=-1).cumsum(-1)
    rows = cdf.shape[0]
    u = uniforms.index_select(0, cursor.view(1)).view(1, 1).expand(rows, 1) * cdf[..., -1:]
    cursor.add_(1)
    return torch.searchsorted(cdf.contiguous(), u.contiguous()).clamp_max_(cdf.shape[-1] - 1).squeeze(-1)


def _param_gb(*modules):
    return sum(p.numel() * p.element_size() for m in modules for p in m.parameters()) / 1e9


def _report(self, frames, gpu_ms):
    decoder = self.model.audio_decoder
    lines = [f"MiniMax Music3 profile: {frames} frames, each section replayed as its own XPU graph "
             "(GPU time, no launch overhead):"]
    for name, seconds in gpu_ms.items():
        lines.append(f"    {name:<10} {1000 * seconds / frames:8.2f} ms")
    lines.append(f"    {'total':<10} {1000 * sum(gpu_ms.values()) / frames:8.2f} ms")
    log.info("\n".join(lines))  # timings first, so a shape-report bug can't eat them
    try:
        head = getattr(self.model, "lm_head_pruned", None) if self.model.pruned_lm_head else getattr(self.model, "lm_head", None)
        log.info(
            f"  shapes: codebooks={self.num_codebooks} depth_layers={len(decoder.layers)} "
            f"main_layers={len(self.model.layers)} hidden={self.model.config.hidden_size} "
            f"audio_vocab={self.audio_vocab_size} pruned_head={bool(self.model.pruned_lm_head)}\n"
            f"  weights: depth_decoder={_param_gb(decoder):.2f} GB  "
            f"audio_extra_embedding={_param_gb(self.model.audio_extra_embedding):.2f} GB  "
            f"main_layers={_param_gb(self.model.layers):.2f} GB  "
            f"c0_head={_param_gb(head) if head is not None else 0:.2f} GB  "
            f"dtype={next(self.model.layers.parameters()).dtype}")
    except Exception as exc:
        log.warning("profile shape report failed: %s", exc)



def _generate(self, input_ids, seed, max_audio_frames, device, cfg_scale=ar.CFG_SCALE, top_k=ar.CFG_TOP_K):
    if MODE == "stock" or not common.is_xpu(device) or not hasattr(torch.xpu, "XPUGraph"):
        return _orig_generate(self, input_ids, seed, max_audio_frames, device, cfg_scale, top_k)

    prompt_tokens = int(input_ids.shape[1])
    if prompt_tokens > ar.MAX_PROMPT_TOKENS:
        raise ValueError(f"MiniMax Music3 prompt has {prompt_tokens} tokens; maximum is {ar.MAX_PROMPT_TOKENS}")

    input_ids = input_ids.to(device)
    execution_dtype = torch.bfloat16 if comfy.model_management.should_use_bf16(device) else torch.float32
    unconditioned = input_ids.clone()
    unconditioned[:, 1:-2] = SPECIAL_TOKEN_IDS["<|audio_cfg|>"]
    text_ids = torch.cat((input_ids, unconditioned), dim=0)
    if self.model.pruned_embedding:
        text_embeds = self.model.embed_tokens_prefill(text_ids, out_dtype=execution_dtype)
    else:
        text_embeds = self.model.embed_tokens(text_ids, out_dtype=execution_dtype)
    decode_limit = min(int(max_audio_frames), ar.MAX_AUDIO_FRAMES)
    frames_cap = decode_limit + 1
    batch = 2

    with common.fixed_kv_scope():
        past = self.model.init_kv_cache(batch, prompt_tokens + frames_cap, device, execution_dtype)
    if not isinstance(past[0], FixedKV):
        log.warning("FixedKV cache not active; using stock loop.")
        del past, text_embeds
        return _orig_generate(self, input_ids, seed, max_audio_frames, device, cfg_scale, top_k)

    # Per-layer prefetch queues sync streams: illegal in capture, overhead when fully loaded.
    old_prefetch = self.model.prefetch_dynamic_vbars
    self.model.prefetch_dynamic_vbars = False
    nan_warned = False
    try:
        output = self.model(None, embeds=text_embeds, past_key_values=past, dtype=execution_dtype)
        last_hidden = output[0][:, -1].clone()
        del output, text_embeds
        vbar = getattr(self, "dynamic_vbars", {}).get(device)
        if vbar is not None:
            comfy.model_management.reset_cast_buffers()
            vbar.set_watermark(vbar.max_size)
        common.zero_stale(past, device)

        generator = torch.Generator(device=device).manual_seed(ar.derive_seed(seed, "ar"))
        hidden = last_hidden.shape[-1]
        hidden_buf = torch.empty((frames_cap, hidden * self.num_codebooks), dtype=execution_dtype, device=device)
        code_buf = torch.empty((frames_cap,), dtype=torch.long, device=device)
        frame_idx = torch.zeros((), dtype=torch.long, device=device)
        positions = torch.full((batch, 1), prompt_tokens, dtype=torch.long, device=device)
        uniforms = torch.rand((frames_cap * self.num_codebooks,), generator=generator, device=device)
        cursor = torch.zeros((), dtype=torch.long, device=device)
        vocab_mask = None
        if not self.model.pruned_lm_head:
            vocab_mask = torch.ones(self.model.vocab_size, dtype=torch.bool, device=device)
            vocab_mask[AUDIO_CODE_OFFSET:AUDIO_CODE_OFFSET + ar.C0_VOCAB_SIZE] = False
            vocab_mask[SPECIAL_TOKEN_IDS["<|audio_end|>"]] = False
        stop = {}
        io = {}

        def part_c0():
            c0, code_or_stop, stop["token"] = self._sample_c0(last_hidden, cfg_scale, top_k, generator, vocab_mask)
            code_buf.index_copy_(0, frame_idx.view(1), code_or_stop.view(1))
            io["c0"] = c0.repeat(2)
            io["c0_embed"] = self._embed_c0(io["c0"], execution_dtype)

        def part_depth():
            codes, depth_hidden = self._depth_codes(last_hidden, io["c0"], io["c0_embed"], generator,
                                                    execution_dtype, cfg_scale, top_k)
            frame_hidden = torch.cat((last_hidden[:1], depth_hidden.to(last_hidden.dtype)), dim=-1)
            hidden_buf.index_copy_(0, frame_idx.view(1), frame_hidden.to(hidden_buf.dtype))
            frame_idx.add_(1)
            io["codes"] = codes

        def part_embed():
            io["feedback"] = self._embed_audio_frame(io["codes"], execution_dtype)

        def part_main():
            out = self.model(None, embeds=io["feedback"], past_key_values=past, dtype=execution_dtype,
                             position_ids=positions)
            last_hidden.copy_(out[0][:, -1])
            positions.add_(1)

        parts = {"c0_sample": part_c0, "depth": part_depth, "embed": part_embed, "main_lm": part_main}

        def step_fn():
            for part in parts.values():
                part()

        progress = comfy.utils.ProgressBar(decode_limit)
        stream = torch.xpu.Stream(device=device)
        stream.wait_stream(torch.xpu.current_stream(device))
        graph = None
        capture_tried = MODE in ("nograph", "profile")
        section_graphs = None
        gpu_time = defaultdict(float)
        profiled = 0
        capture_at = max(1, int(CAPTURE_AT))
        checked, stop_at = 0, None
        _SAMPLER.state = (uniforms, cursor)
        with torch.xpu.stream(stream):
            for frame_index in comfy.utils.model_trange(frames_cap, desc="AR sampling"):
                comfy.model_management.throw_exception_if_processing_interrupted()
                if graph is None and not capture_tried and frame_index >= capture_at:
                    capture_tried = True
                    graph = common.capture(step_fn, stream, "MiniMax Music3")
                    if graph is not None:
                        log.info("MiniMax Music3: XPU graph captured at frame %d", frame_index)
                if MODE == "profile" and section_graphs is None and frame_index >= capture_at:
                    # Capture each part separately; tensors in `io` stay alive, so later graphs
                    # read the buffers earlier graphs write. Replays then give per-part GPU time.
                    section_graphs = {}
                    for name, part in parts.items():
                        g = common.capture(part, stream, f"MiniMax Music3 {name}")
                        if g is None:
                            raise RuntimeError(f"profile: could not capture section {name}")
                        section_graphs[name] = g
                    log.info("MiniMax Music3: profiling %d frames with per-section graphs", PROFILE_FRAMES)
                if section_graphs is not None:
                    for name, g in section_graphs.items():
                        torch.xpu.synchronize()
                        start = time.perf_counter()
                        g.replay()
                        torch.xpu.synchronize()
                        gpu_time[name] += time.perf_counter() - start
                    profiled += 1
                    if profiled == PROFILE_FRAMES:
                        _report(self, PROFILE_FRAMES, gpu_time)
                        stop_at = frame_index + 1
                        break
                elif graph is not None:
                    graph.replay()
                else:
                    step_fn()
                if (frame_index + 1) % CHECK_EVERY == 0 or frame_index + 1 == frames_cap:
                    if not nan_warned and not bool(torch.isfinite(last_hidden).all()):
                        nan_warned = True
                        log.warning("MiniMax Music3: non-finite hidden state at frame %d", frame_index)
                    for i, code in enumerate(code_buf[checked:frame_index + 1].tolist(), start=checked):
                        if code == stop["token"]:
                            stop_at = i
                            break
                    checked = frame_index + 1
                    progress.update_absolute(min(checked, decode_limit))
                    if stop_at is not None:
                        break
        torch.xpu.current_stream(device).wait_stream(stream)
        del graph, section_graphs
    finally:
        _SAMPLER.state = None
        self.model.prefetch_dynamic_vbars = old_prefetch
        comfy.model_prefetch.cleanup_prefetch_queues()

    # Stock never keeps frame 0, and drops the frame whose c0 is the stop token.
    end = stop_at if stop_at is not None else frames_cap
    frames = hidden_buf[1:end][:decode_limit]
    if frames.shape[0] == 0:
        raise ValueError("MiniMax Music3 generated zero audio frames")
    return frames.to(device="cpu")


def install():
    common.install_attention()
    ar.sample_topk = _sample_topk
    ar.MiniMaxMusic3AR.generate = _generate


def uninstall():
    ar.sample_topk = _orig_sample_topk
    ar.MiniMaxMusic3AR.generate = _orig_generate