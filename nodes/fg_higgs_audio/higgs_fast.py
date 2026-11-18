"""
Faster decode loop for Higgs v3, after fg_fish_audio/fish_s2_fast.py.

Every audio frame (25 per second of speech) is one step of the 36-layer Qwen3 backbone at batch 1, then
8 codebooks sampled at once. The eager loop (higgs_core.speak, mode "eager") leaves a lot on the table:

1. transformers' DynamicCache grows (concatenates) every layer's K / V every frame, and the forward
   rebuilds masks and position tensors on the host each call.
2. The sampler reads codes[0] back to the host every frame (`.item()`) to watch for end-of-speech: a
   sync per frame.
3. ~600 small kernel launches per frame.

This loop runs the backbone's own weights on a fixed-size K / V cache (one slot per position, a
device-side length mask), samples on the device with uniforms drawn up front, applies the delay
pattern from a device-side step counter, and checks for end-of-speech every CHECK_EVERY frames. That
makes one frame graph-safe, so in mode "graph" it is captured once as an XPU graph and replayed.

MODE "eager"   higgs_core's own loop (transformers forward + DynamicCache)
MODE "nograph" this loop, eager
MODE "graph"   this loop + XPU graph capture after CAPTURE_AT warm-up frames (XPU only; elsewhere nograph)

Sampling is the same as the eager loop's (temperature, then top-k, then top-p on the tempered softmax),
drawn by inverse CDF from pre-drawn uniforms, so a seed gives a different take than mode "eager" but
from the same distribution.
"""
import contextlib
import logging

import torch
import torch.nn.functional as F

import comfy.model_management as mm

log = logging.getLogger("FG_Higgs_XPU")

MODES = ("graph", "nograph", "eager")
CAPTURE_AT = 4    # eager warm-up frames before capture (oneDNN / oneMKL initialise lazily)
CHECK_EVERY = 8   # frames between end-of-speech checks (each one is a host sync)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class FastBackbone:
    """The backbone's layers on a fixed-size K / V cache. Built per line (the cache is sized to it)."""

    def __init__(self, backbone, capacity: int, device, dtype):
        cfg = backbone.config
        self.layers = backbone.layers
        self.norm = backbone.norm
        for layer in self.layers:
            if getattr(layer.self_attn, "sliding_window", None):
                raise RuntimeError("Higgs fast loop: sliding-window layers aren't supported")
        a0 = self.layers[0].self_attn
        self.heads, self.kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
        self.head_dim, self.scaling = a0.head_dim, a0.scaling
        self.capacity, self.dev, self.dtype = capacity, device, dtype
        shape = (1, self.kv_heads, capacity, self.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in self.layers]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in self.layers]
        positions = torch.arange(capacity, device=device).view(1, -1)
        cos, sin = backbone.rotary_emb(torch.empty(1, device=device, dtype=dtype), positions)
        self.cos, self.sin = cos[0], sin[0]                    # [capacity, head_dim]
        self.slots = torch.arange(capacity, device=device)

    def _attn(self, i, layer, x, pos, mask):
        """x [1, L, D]; pos [L] (long, device); mask [1, 1, L, capacity] bool or None (prefill: causal)."""
        a = layer.self_attn
        L = x.shape[1]
        q = a.q_norm(a.q_proj(x).view(1, L, self.heads, self.head_dim)).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(1, L, self.kv_heads, self.head_dim)).transpose(1, 2)
        v = a.v_proj(x).view(1, L, self.kv_heads, self.head_dim).transpose(1, 2)
        cos = self.cos.index_select(0, pos).unsqueeze(0).unsqueeze(0)
        sin = self.sin.index_select(0, pos).unsqueeze(0).unsqueeze(0)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        self.k[i].index_copy_(2, pos, k)
        self.v[i].index_copy_(2, pos, v)
        if mask is None:  # prefill: the prompt attends to itself, causally
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scaling,
                                                 enable_gqa=self.heads != self.kv_heads)
        else:
            out = F.scaled_dot_product_attention(q, self.k[i], self.v[i], attn_mask=mask, scale=self.scaling,
                                                 enable_gqa=self.heads != self.kv_heads)
        return a.o_proj(out.transpose(1, 2).reshape(1, L, -1))

    def forward(self, x, pos, mask=None):
        """[1, L, D] -> last position's hidden [D] (after the final norm)."""
        for i, layer in enumerate(self.layers):
            x = x + self._attn(i, layer, layer.input_layernorm(x), pos, mask)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.norm(x)[0, -1]


def _sample(logits, temperature: float, top_p: float, top_k: int, u, rank):
    """[N, V] logits -> [N] codes: temperature, top-k, top-p on the tempered softmax (as higgs_core._sample),
    then an inverse-CDF draw with one pre-drawn uniform per codebook."""
    s, idx = torch.sort(logits / temperature, descending=True, dim=-1)
    if top_k:
        s = s.masked_fill(rank >= top_k, float("-inf"))
    if top_p is not None and top_p < 1.0:
        cum = torch.softmax(s, dim=-1).cumsum(dim=-1)
        remove = torch.cat([torch.zeros_like(cum[:, :1], dtype=torch.bool), cum[:, :-1] > top_p], dim=-1)
        s = s.masked_fill(remove, float("-inf"))
    cdf = torch.softmax(s, dim=-1).cumsum(dim=-1)
    j = torch.searchsorted(cdf, u.view(-1, 1) * cdf[:, -1:]).clamp_max_(s.shape[-1] - 1)
    return idx.gather(-1, j).squeeze(-1)


@torch.inference_mode()
def generate(h, embeds, max_frames: int, seed: int, sampling: dict, mode: str = "graph", capture_at: int = CAPTURE_AT,
             progress=None) -> torch.Tensor:
    """embeds: the prompt [L, D] (text + reference rows, built by higgs_core). Returns delayed frames [F, N]."""
    from .higgs_core import BOC_ID, EOC_ID

    m = h.module
    dev = embeds.device
    n = h.num_codebooks
    T = embeds.shape[0]
    fb = FastBackbone(m.backbone, T + max_frames + 1, dev, h.dtype)
    temperature, top_p, top_k = float(sampling["temperature"]), sampling.get("top_p"), sampling.get("top_k")
    greedy = temperature <= 1e-5
    vocab = m.embedding.vocab_size
    rank = torch.arange(vocab, device=dev).view(1, -1)
    cb = torch.arange(n, device=dev)

    gen = torch.Generator(device=dev).manual_seed(int(seed))
    uniforms = torch.rand((max_frames, n), device=dev, generator=gen, dtype=torch.float32)
    out = torch.empty((max_frames, n), device=dev, dtype=torch.long)

    def pick(hidden, step):
        logits = m.embedding.head(hidden.view(1, -1)).squeeze(0).float()         # [N, V]
        if greedy:
            codes = logits.argmax(dim=-1)
        else:
            codes = _sample(logits, temperature, top_p, top_k, uniforms.index_select(0, step.view(1)).view(-1, 1), rank)
        return torch.where(cb > step, torch.full_like(codes, BOC_ID), codes)       # the delay pattern's first N frames

    # prefill (variable length: always eager), then frame 0
    hidden = fb.forward(embeds.unsqueeze(0).to(h.dtype), torch.arange(T, device=dev))
    step_index = torch.zeros((), device=dev, dtype=torch.long)
    first = pick(hidden, step_index)
    out[0] = first
    cur = first.clone()
    pos = torch.full((1,), T, device=dev, dtype=torch.long)
    step_index.add_(1)

    def step_fn():
        x = m.embedding(cur.view(1, -1)).view(1, 1, -1).to(h.dtype)
        mask = (fb.slots <= pos).view(1, 1, 1, -1)
        hid = fb.forward(x, pos, mask)
        codes = pick(hid, step_index)
        out.index_copy_(0, step_index.view(1), codes.view(1, -1))
        cur.copy_(codes)
        pos.add_(1)
        step_index.add_(1)

    use_graph = mode == "graph" and dev.type == "xpu" and hasattr(torch.xpu, "XPUGraph")
    if mode == "graph" and not use_graph:
        log.info("Higgs: graph mode needs XPU; running the fast loop eagerly")
    if dev.type == "xpu":
        stream = torch.xpu.Stream(device=dev)
        stream.wait_stream(torch.xpu.current_stream(dev))
        scope = torch.xpu.stream(stream)
    else:
        stream, scope = None, contextlib.nullcontext()

    # end of speech: codes[0] == EOC once the delay is over (frame >= N), then N - 2 more frames
    graph, tried = None, not use_graph
    n_frames, checked = max_frames, n
    try:
        with scope:
            for step in range(1, max_frames):
                mm.throw_exception_if_processing_interrupted()
                if graph is None and not tried and step >= max(1, int(capture_at)):
                    tried = True
                    from ..fg_yue2 import _xpu_ar_common as common
                    graph = common.capture(step_fn, stream, "Higgs")
                    if graph is not None:
                        log.info("Higgs: XPU graph captured at frame %d", step)
                if graph is not None:
                    graph.replay()
                else:
                    step_fn()
                done = step + 1
                if done % CHECK_EVERY == 0 or done == max_frames:
                    if progress:
                        progress.update(done)
                    if done > checked:
                        firsts = out[checked:done, 0].tolist()
                        if EOC_ID in firsts:
                            eoc = checked + firsts.index(EOC_ID)
                            n_frames = min(max_frames, eoc + 1 + max(0, n - 2))
                            if n_frames <= done:
                                break
                            checked = max_frames  # found: keep going only until the countdown frames exist
                        else:
                            checked = done
                    if done >= n_frames:
                        break
    finally:
        if stream is not None:
            torch.xpu.current_stream(dev).wait_stream(stream)
        del graph
    if progress:
        progress.done(n_frames)
    if n_frames == max_frames:
        log.warning("Higgs: reached the frame budget (%d) before the end of speech", max_frames)
    return out[:n_frames].clone()
