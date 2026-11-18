"""
Faster decode loop for Fish S2 Pro, after fg_yue2/yue2_xpu_patch.py.

Fish is Dual-AR: every audio frame (~21.5 per second of speech) is one step of the 36-layer slow
transformer, then 10 sequential steps of the 4-layer fast transformer (one per codebook), all at
batch 1. The stock loop (fish_speech/models/text2semantic/inference.py) leaves a lot on the table:

1. Attention repeats the 8 KV heads to 32 over the WHOLE fixed-size cache, in every layer, every
   frame (`repeat_interleave`): at max_seq_len 8192 that is ~64 MB copied per layer, twice.
   SDPA's enable_gqa does the same maths without the copies.
2. The main token is sampled twice per frame (normal + the RAS high-temperature fallback), each
   time sorting all 155,776 vocabulary logits, although a bias then leaves only the 4,096 semantic
   tokens + <|im_end|> possible. Sampling those 4,097 gives the same distribution.
3. Each step builds small tensors on the host (fast-transformer positions) and masks with boolean
   indexing (a host sync), which also rules out graph capture.
4. ~1,000 tiny kernel launches per frame. With the step made graph-safe, it is captured once as an
   XPU graph and replayed for every remaining frame (graph mode).

MODE "stock"   fish-speech's own loop, untouched
MODE "nograph" this loop (1-3), eager
MODE "graph"   this loop + XPU graph capture after CAPTURE_AT eager warm-up frames (XPU only;
               anywhere else it runs as nograph)

Sampling draws its uniforms up front (inverse CDF), as the YuE2 patch does, so a seed does not
reproduce the stock loop's output, but the distribution is the same: same top-k / top-p / temperature
and the same repetition-aware fallback (RAS).
"""

import contextlib
import logging
import math

import torch
import torch.nn.functional as F

import comfy.model_management as mm

log = logging.getLogger("FG_FishS2_XPU")

MODES = ("graph", "nograph", "stock")
CAPTURE_AT = 4         # eager warm-up frames before capture (oneDNN / oneMKL initialise lazily)
CHECK_EVERY = 8        # frames between end-of-speech checks (each one is a host sync)
RAS_WIN, RAS_TEMP, RAS_TOP_P = 10, 1.0, 0.9   # fish-speech's repetition-aware sampling constants


# ---------------------------------------------------------------- progress

# ---------------------------------------------------------------- 1. attention without the KV copies

_orig_attention = None


def _attention_forward(self, x, freqs_cis, mask, input_pos=None):
    """llama.Attention.forward with enable_gqa instead of repeat_interleave (slow layers, cached path)."""
    if not self.use_sdpa or mask is None or self.kv_cache is None:
        return _orig_attention(self, x, freqs_cis, mask, input_pos)
    from fish_speech.models.text2semantic.llama import apply_rotary_emb

    bsz, seqlen, _ = x.shape
    q_size, kv_size = self.n_head * self.head_dim, self.n_local_heads * self.head_dim
    q, k, v = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)
    q = q.view(bsz, seqlen, self.n_head, self.head_dim)
    k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    if self.attention_qk_norm:
        q, k = self.q_norm(q), self.k_norm(k)
    q, k = apply_rotary_emb(q, freqs_cis), apply_rotary_emb(k, freqs_cis)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    k, v = self.kv_cache.update(input_pos, k, v)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=self.n_head != self.n_local_heads)
    return self.wo(y.transpose(1, 2).contiguous().view(bsz, seqlen, q_size))


def set_attention(patched: bool) -> None:
    global _orig_attention
    from fish_speech.models.text2semantic import llama
    if _orig_attention is None:
        _orig_attention = llama.Attention.forward
    llama.Attention.forward = _attention_forward if patched else _orig_attention


# ---------------------------------------------------------------- 2-4. the loop

class Decoder:
    """Per-model constants; built once per device and kept on the FishS2 handle."""

    def __init__(self, lm, device):
        from fish_speech.tokenizer import IM_END_TOKEN

        cfg = lm.config
        self.lm, self.cfg, self.dev = lm, cfg, device
        self.ncb, self.cb = cfg.num_codebooks, cfg.codebook_size
        self.sem_b, self.sem_e = cfg.semantic_begin_id, cfg.semantic_end_id
        self.im_end = lm.tokenizer.get_token_id(IM_END_TOKEN)
        self.allowed = torch.cat([torch.arange(self.sem_b, self.sem_e + 1), torch.tensor([self.im_end])]).to(device)
        table = lm.embeddings.weight if cfg.tie_word_embeddings else lm.output.weight
        self.head = table.index_select(0, self.allowed)           # [4097, dim]: the only rows that can be picked
        self.offsets = (torch.arange(self.ncb, device=device) * self.cb).view(1, self.ncb, 1)
        self.fast_pos = [torch.tensor([i], device=device, dtype=torch.long) for i in range(self.ncb)]
        self.rank_main = torch.arange(self.allowed.numel(), device=device)
        self.rank_fast = torch.arange(self.cb, device=device)
        self.scale = 1.0 / math.sqrt(self.ncb + 1)

    @staticmethod
    def _sample(logits, temperature, top_p, top_k, u, rank):
        """fish-speech's logits_to_probs + sampling, on one row: top-k / top-p from the untempered
        softmax, then temperature, then an inverse-CDF draw with the pre-drawn uniform u."""
        s, idx = torch.sort(logits, descending=True)
        cum = torch.softmax(s, dim=-1).cumsum(dim=-1)
        remove = ((cum > top_p) | (rank >= top_k)) & (rank > 0)
        s = torch.where(remove, float("-inf"), s) / temperature
        cdf = torch.softmax(s, dim=-1).cumsum(dim=-1)
        j = torch.searchsorted(cdf, (u * cdf[-1]).view(1)).clamp_max_(s.numel() - 1)
        return idx.index_select(0, j)

    def frame(self, main_logits, hidden, u, prev, p):
        """Main token (+ RAS fallback when prev is given) then the fast transformer's 9 codebooks.
        Returns [ncb + 1] long: main token, then codebooks 0..ncb-1."""
        lm = self.lm
        j = self._sample(main_logits, p["temperature"], p["top_p"], p["top_k"], u[0], self.rank_main)
        tok = self.allowed.index_select(0, j)
        if prev is not None:
            j_hi = self._sample(main_logits, RAS_TEMP, RAS_TOP_P, p["top_k"], u[1], self.rank_main)
            tok_hi = self.allowed.index_select(0, j_hi)
            repeat = (prev == tok).any() & (tok >= self.sem_b) & (tok <= self.sem_e)
            tok = torch.where(repeat, tok_hi, tok)
        lm.forward_generate_fast(hidden, self.fast_pos[0])
        a = (tok - self.sem_b).clamp(0, self.cb - 1)
        codes = [tok, a]
        h = lm.fast_embeddings(a)
        for i in range(1, self.ncb):
            logits = lm.forward_generate_fast(h, self.fast_pos[i]).view(-1).float()
            a = self._sample(logits, p["temperature"], p["top_p"], p["top_k"], u[1 + i], self.rank_fast)
            h = lm.fast_embeddings(a)
            codes.append(a)
        return torch.cat(codes)

    def slow(self, cur, pos):
        """BaseTransformer.forward_generate for one frame, without boolean indexing or full-vocab logits."""
        lm, cfg = self.lm, self.cfg
        tok = cur[:, 0]                                                    # [1, 1]
        vq = lm.codebook_embeddings(cur[:, 1:] + self.offsets).sum(dim=1)   # [1, 1, dim]
        is_vq = ((tok >= self.sem_b) & (tok <= self.sem_e)).unsqueeze(-1)   # [1, 1, 1]
        x = lm.embeddings(tok) + vq * is_vq
        if cfg.scale_codebook_embeddings:
            x = torch.where(is_vq, x * self.scale, x)
        mask = lm.causal_mask[None, None, pos, : lm.max_seq_len]
        freqs = lm.freqs_cis[pos]
        for layer in lm.layers:
            x = layer(x, freqs, mask, input_pos=pos)
        slow_out = lm.norm(x)
        main_logits = (slow_out.view(1, -1) @ self.head.T).view(-1).float()
        hidden = slow_out if getattr(cfg, "norm_fastlayer_input", False) else x
        return main_logits, lm.fast_project_in(hidden)


@torch.inference_mode()
def generate(decoder: Decoder, prompt, audio_masks, audio_parts, max_new_tokens: int, seed: int, sampling: dict,
             mode: str = "graph", capture_at: int = CAPTURE_AT, progress=None):
    """Prefill eagerly, then decode frame by frame. Returns codes [ncb, frames] (without <|im_end|>)."""
    lm, dev, ncb = decoder.lm, decoder.dev, decoder.ncb
    T = prompt.size(1)
    budget = lm.max_seq_len - T
    max_frames = min(int(max_new_tokens), budget) if max_new_tokens else budget
    if max_frames < 1:
        raise ValueError("Fish S2: no room left in the context for this line")
    p = {"temperature": float(sampling["temperature"]), "top_p": float(sampling["top_p"]), "top_k": int(sampling["top_k"])}

    gen = torch.Generator(device=dev).manual_seed(int(seed))
    uniforms = torch.rand((max_frames, ncb + 1), device=dev, generator=gen, dtype=torch.float32)
    out = torch.empty((ncb + 1, max_frames), device=dev, dtype=torch.long)

    # prefill (variable length: always eager), first frame without RAS like fish-speech
    res = lm.forward_generate(prompt.view(1, ncb + 1, -1), torch.arange(T, device=dev), audio_masks, audio_parts)
    first = decoder.frame(res.logits[0, -1].index_select(0, decoder.allowed).float(), res.hidden_states, uniforms[0], None, p)
    del res
    out[:, 0] = first
    if int(first[0]) == decoder.im_end:
        if progress:
            progress.done(0)
        return out[1:, :0]

    cur = first.view(1, ncb + 1, 1).clone()
    pos = torch.tensor([T], device=dev, dtype=torch.long)
    step_index = torch.ones((), device=dev, dtype=torch.long)
    prev = torch.zeros((1, RAS_WIN), device=dev, dtype=torch.long)

    def step_fn():
        u = uniforms.index_select(0, step_index.view(1)).view(-1)
        main_logits, hidden = decoder.slow(cur, pos)
        codes = decoder.frame(main_logits, hidden, u, prev, p)
        out.index_copy_(1, step_index.view(1), codes.view(-1, 1))
        prev.copy_(torch.cat([prev[:, 1:], codes[:1].view(1, 1)], dim=1))
        cur.copy_(codes.view(1, -1, 1))
        pos.add_(1)
        step_index.add_(1)

    use_graph = mode == "graph" and dev.type == "xpu" and hasattr(torch.xpu, "XPUGraph")
    if mode == "graph" and not use_graph:
        log.info("Fish S2: graph mode needs XPU; running the patched loop eagerly")
    if dev.type == "xpu":
        stream = torch.xpu.Stream(device=dev)
        stream.wait_stream(torch.xpu.current_stream(dev))
        scope = torch.xpu.stream(stream)
    else:
        stream, scope = None, contextlib.nullcontext()

    graph, tried = None, not use_graph
    n_frames, checked = max_frames, 1
    try:
        with scope:
            for step in range(1, max_frames):
                mm.throw_exception_if_processing_interrupted()
                if graph is None and not tried and step >= max(1, int(capture_at)):
                    tried = True
                    from ..fg_yue2 import _xpu_ar_common as common
                    graph = common.capture(step_fn, stream, "Fish S2")
                    if graph is not None:
                        log.info("Fish S2: XPU graph captured at frame %d", step)
                if graph is not None:
                    graph.replay()
                else:
                    step_fn()
                if (step + 1) % CHECK_EVERY == 0 or step + 1 == max_frames:
                    if progress:
                        progress.update(step + 1)
                    mains = out[0, checked: step + 1].tolist()
                    if decoder.im_end in mains:
                        n_frames = checked + mains.index(decoder.im_end)
                        break
                    checked = step + 1
    finally:
        if stream is not None:
            torch.xpu.current_stream(dev).wait_stream(stream)
        del graph
    if progress:
        progress.done(n_frames)
    if n_frames == max_frames:
        log.warning("Fish S2: reached the frame budget (%d) before the end of speech", max_frames)
    return out[1:, :n_frames].clone()
