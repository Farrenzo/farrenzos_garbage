"""Higgs Audio v3 TTS 4B (bosonai/higgs-audio-v3-tts-4b) inside ComfyUI, on XPU / CUDA.

Ported from Renzo Studio's backend/app/services/tts/higgs_engine.py (itself after
.planning/audio_plan/run_higgs_tts_xpu.py). The checkpoint holds three things in one file: a plain
Qwen3-4B backbone ("body.*", "tied.embedding.text_embedding"), a fused 8-codebook embedding / head
("tied.embedding.modality_embeddings.0.embedding", tied), and the Higgs v2 audio codec
("tied.embedding.modality_embeddings.0.model.*", its config in higgs-audio-v2-3B/tokenizer).

* Loading streams the checkpoint once onto a meta-device skeleton, to the CPU; backbone, embedding and
  codec sit in one module behind a ModelPatcher, so ComfyUI moves them on / off the GPU like any model.
* The codec stays fp32 (its decoder's ConvTranspose is unstable in bf16).
* Prompt (one line per call):
      <|tts|> [<|ref_text|> transcripts] <|ref_audio|> [-100] x N <|text|> line <|audio|>
  N = the delayed reference-code rows. Several references (the voice sample, then the speaker's own
  earlier lines) are concatenated in order into that one slot: Higgs has no multi-speaker turns, so a
  speaker only ever hears itself (AUDIO_SPEC §6.7.2).

License: the Higgs Audio v3 weights are under Boson AI's license (see the checkpoint's LICENSE).
"""
import hashlib
import itertools
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.model_patcher

log = logging.getLogger("FG_Higgs")
SAMPLE_RATE = 24_000
FRAMES_PER_S = 25
BOC_ID = 1024           # begin-of-codebook padding, inside the per-codebook vocab
EOC_ID = 1025           # end-of-codebook marker
AUDIO_PLACEHOLDER_ID = -100
QWEN3_ROPE_THETA = 1_000_000  # Higgs ships rope_theta=null; HF's default (10000) is wrong
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

_CODEC_PREFIX = "tied.embedding.modality_embeddings.0.model."
_EMBED_PREFIX = "tied.embedding.modality_embeddings.0.embedding."
_HEAD_PREFIX = "tied.head.modality_heads.0."
_BACKBONE_PREFIX_MAP = {"tied.embedding.text_embedding.": "embed_tokens.", "body.layers.": "layers.", "body.norm.": "norm."}
_REQUIRED_SPECIALS = ("<|tts|>", "<|ref_audio|>", "<|text|>", "<|audio|>")
TAG = re.compile(r"<\|([a-z_]+):([a-z_]+)\|>")


# ---------------------------------------------------------------- fused embedding, delay pattern

class FusedEmbedding(torch.nn.Module):
    """One [num_codebooks * vocab, D] table; per-codebook lookups summed. Also the (tied) head."""

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(num_codebooks * vocab_size, hidden_size))
        self.num_codebooks, self.vocab_size = num_codebooks, vocab_size

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.num_codebooks, device=codes.device, dtype=codes.dtype) * self.vocab_size
        return F.embedding(codes + offsets, self.weight).sum(dim=-2)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        """[L, D] -> [L, num_codebooks, vocab] logits (tie_word_embeddings)."""
        return F.linear(hidden, self.weight).reshape(hidden.shape[0], self.num_codebooks, self.vocab_size)


def apply_delay_pattern(codes_TN: torch.Tensor) -> torch.Tensor:
    """[T, N] raw codes -> [T + N - 1, N] delayed, BOC / EOC padded."""
    T, N = codes_TN.shape
    out = torch.full((T + N - 1, N), EOC_ID, dtype=codes_TN.dtype, device=codes_TN.device)
    t_idx = torch.arange(T + N - 1, device=codes_TN.device)
    for c in range(N):
        out[t_idx < c, c] = BOC_ID
        out[c: c + T, c] = codes_TN[:, c]
    return out


def reverse_delay_pattern(delayed: torch.Tensor) -> torch.Tensor:
    """[L, N] delayed -> [L - (N - 1), N] raw."""
    length, n = delayed.shape
    rows = length - (n - 1)
    if rows <= 0:
        return delayed.new_empty((0, n))
    out = torch.empty((rows, n), dtype=delayed.dtype, device=delayed.device)
    for c in range(n):
        out[:, c] = delayed[c: c + rows, c]
    return out


# ---------------------------------------------------------------- sampler (delay / EOC state machine)

@dataclass
class SamplerState:
    num_codebooks: int
    delay_count: int = 0
    eoc_countdown: int | None = None
    done: bool = False
    last_codes: torch.Tensor | None = None


def _sample(logits_NV: torch.Tensor, temperature: float, top_p: float | None, top_k: int | None) -> torch.Tensor:
    if temperature <= 1e-5:
        return logits_NV.argmax(dim=-1)
    logits = logits_NV / temperature
    if top_k:
        kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
        logits = torch.where(logits < kth, float("-inf"), logits)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        remove = sorted_logits.softmax(dim=-1).cumsum(dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        logits = torch.where(torch.zeros_like(remove).scatter_(-1, sorted_idx, remove), float("-inf"), logits)
    return logits.softmax(dim=-1).multinomial(num_samples=1).squeeze(-1)


def sampler_step(logits_NV: torch.Tensor, st: SamplerState, temperature: float, top_p, top_k) -> torch.Tensor:
    n = st.num_codebooks
    if st.done:
        return torch.full((n,), -1, dtype=torch.long, device=logits_NV.device)
    codes = _sample(logits_NV, temperature, top_p, top_k).to(torch.long)
    if st.delay_count < n:
        if st.delay_count + 1 < n:
            codes[st.delay_count + 1:] = BOC_ID
        st.delay_count += 1
    elif st.eoc_countdown is not None:
        st.eoc_countdown -= 1
        if st.eoc_countdown <= 0:
            st.done = True
    elif int(codes[0].item()) == EOC_ID:
        if n <= 2:
            st.done = True
        else:
            st.eoc_countdown = n - 2
    if not st.done:
        st.last_codes = codes.clone()
    return codes


# ---------------------------------------------------------------- loading

def _route(name: str) -> tuple[str, str] | None:
    if name.startswith(_CODEC_PREFIX):
        return "codec", name[len(_CODEC_PREFIX):]
    if name.startswith(_EMBED_PREFIX):
        return "embedding", name[len(_EMBED_PREFIX):]
    if name.startswith(_HEAD_PREFIX):
        return None  # tied to the embedding
    for src, dst in _BACKBONE_PREFIX_MAP.items():
        if name.startswith(src):
            return "backbone", dst + name[len(src):]
    return None  # e.g. the untied text head: not used for TTS


def _shards(ckpt: Path) -> list[Path]:
    index = ckpt / "model.safetensors.index.json"
    if index.is_file():
        return [ckpt / n for n in sorted(set(json.loads(index.read_text())["weight_map"].values()))]
    return [ckpt / "model.safetensors"]


BUNDLED_CODEC_CONFIG = Path(__file__).with_name("codec_config.json")  # bosonai/higgs-audio-v2-tokenizer's config


def find_codec_config(ckpt: Path) -> Path | None:
    """The Higgs v2 codec config: one next to the checkpoint if there is one, else the copy in this package
    (the model's own config.json doesn't carry it)."""
    for base in (ckpt, ckpt.parent):
        for c in (base / "higgs-audio-v2-3B" / "tokenizer" / "config.json", base / "tokenizer" / "config.json",
                  base / "codec_config.json"):
            if c.is_file():
                return c
    return BUNDLED_CODEC_CONFIG if BUNDLED_CODEC_CONFIG.is_file() else None


class HiggsModule(torch.nn.Module):
    """Backbone + fused embedding / head + codec in one module, so one ModelPatcher moves them all."""

    def __init__(self, backbone, embedding, codec):
        super().__init__()
        self.backbone, self.embedding, self.codec = backbone, embedding, codec


@dataclass
class Higgs:
    patcher: object
    tokenizer: object
    special: dict
    num_codebooks: int
    dtype: torch.dtype
    path: str
    work_bytes: int
    codes_cache: dict = field(default_factory=dict)

    @property
    def module(self) -> HiggsModule:
        return self.patcher.model


def load(path: str, codec_config: str = "", precision: str = "bfloat16") -> Higgs:
    """Backbone + codec on the CPU (ComfyUI moves them to the GPU when a line is spoken)."""
    from safetensors import safe_open
    from transformers import AutoTokenizer, HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel, Qwen3Config, Qwen3Model
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    t0 = time.time()
    ckpt = Path(path)
    if not (ckpt / "config.json").is_file():
        raise FileNotFoundError(f"Higgs: config.json not found in {ckpt}")
    ccfg_path = Path(codec_config) if codec_config else find_codec_config(ckpt)
    if not ccfg_path or not ccfg_path.is_file():
        raise FileNotFoundError("Higgs: no codec config found (fg_higgs_audio/codec_config.json is missing); "
                                "set codec_config on the loader")
    dtype = DTYPES[precision]
    config = json.loads((ckpt / "config.json").read_text())
    text_cfg = dict(config["text_config"])
    text_cfg.pop("model_type", None)
    if text_cfg.get("rope_theta") is None:
        text_cfg["rope_theta"] = QWEN3_ROPE_THETA
    qcfg = Qwen3Config(**text_cfg)
    qcfg._attn_implementation = "sdpa"
    acfg = config["audio_encoder_config"]
    num_codebooks = int(acfg["num_codebooks"])
    vocab, hidden = int(acfg["vocab_size"]), int(acfg.get("out_dim", qcfg.hidden_size))

    with torch.device("meta"):
        backbone = Qwen3Model(qcfg)
        embedding = FusedEmbedding(num_codebooks, vocab, hidden)

    buckets: dict[str, dict] = {"backbone": {}, "embedding": {}, "codec": {}}
    for shard in _shards(ckpt):
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                routed = _route(key)
                if routed is None:
                    continue
                bucket, name = routed
                t = f.get_tensor(key)
                if t.is_floating_point():
                    t = t.float() if bucket == "codec" else t.to(dtype)
                buckets[bucket][name] = t
                del t
    missing, unexpected = backbone.load_state_dict(buckets["backbone"], strict=False, assign=True)
    missing = [m for m in missing if "rotary_emb" not in m]
    if missing:
        raise RuntimeError(f"Higgs: backbone weights missing: {missing[:8]}")
    if unexpected:
        log.info("Higgs: %d backbone tensors not used, e.g. %s", len(unexpected), unexpected[:5])
    embedding.load_state_dict(buckets["embedding"], assign=True)
    buckets["backbone"].clear()
    buckets["embedding"].clear()
    # rotary inv_freq is a non-persistent buffer: still on meta after assign-loading
    for name, mod in list(backbone.named_modules()):
        if type(mod).__name__.endswith("RotaryEmbedding") and any(b.is_meta for b in mod.buffers(recurse=False)):
            parent, _, attr = name.rpartition(".")
            setattr(backbone.get_submodule(parent) if parent else backbone, attr, Qwen3RotaryEmbedding(config=qcfg))
    for mod_name, mod in (("backbone", backbone), ("embedding", embedding)):
        left = [n for n, t in itertools.chain(mod.named_parameters(), mod.named_buffers()) if t.is_meta]
        if left:
            raise RuntimeError(f"Higgs: {mod_name}: {len(left)} tensors still on meta, e.g. {left[:4]}")

    codec = HiggsAudioV2TokenizerModel(HiggsAudioV2TokenizerConfig.from_json_file(str(ccfg_path)))
    miss, _ = codec.load_state_dict(buckets["codec"], strict=False)
    if len(miss) > len(buckets["codec"]) // 2:
        raise RuntimeError(f"Higgs: codec weights didn't match ({len(miss)} missing)")
    buckets.clear()
    codec = codec.float()

    module = HiggsModule(backbone, embedding, codec).eval()
    for p in module.parameters():
        p.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), local_files_only=True)
    special = tokenizer.get_added_vocab()
    if any(t not in special for t in _REQUIRED_SPECIALS):
        raise ValueError(f"Higgs: the tokenizer lacks the TTS specials {_REQUIRED_SPECIALS}")
    patcher = comfy.model_patcher.CoreModelPatcher(module, load_device=mm.get_torch_device(),
                                                   offload_device=mm.unet_offload_device())
    log.info("Higgs loaded from %s in %.1f s", ckpt, time.time() - t0)
    return Higgs(patcher, tokenizer, special, num_codebooks, dtype, str(ckpt), work_bytes=1024 * 2**20)


# ---------------------------------------------------------------- audio <-> codes

def _mono(audio: dict) -> torch.Tensor:
    """ComfyUI AUDIO -> mono float32 [T] at 24 kHz."""
    wav = audio["waveform"][0].float()
    wav = wav.mean(dim=0) if wav.dim() == 2 else wav
    sr = int(audio["sample_rate"])
    if sr != SAMPLE_RATE:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


@torch.inference_mode()
def encode(h: Higgs, audio: dict) -> torch.Tensor:
    """AUDIO -> raw codes [T, N] (CPU), cached by content (a take loaded from disk is encoded once)."""
    wav = _mono(audio)
    if wav.numel() < SAMPLE_RATE:
        wav = F.pad(wav, (0, SAMPLE_RATE - wav.numel()))
    key = hashlib.sha1(wav.numpy().tobytes()).hexdigest()
    if key in h.codes_cache:
        return h.codes_cache[key]
    codec = h.module.codec
    dev = next(codec.parameters()).device
    codes = codec.encode(wav.view(1, 1, -1).to(dev)).audio_codes.squeeze(0).transpose(0, 1).to(torch.long).cpu()
    h.codes_cache[key] = codes
    return codes


def unknown_tags(text: str) -> list[str]:
    """Tags Higgs doesn't know are read aloud or degrade the line; the node warns about them."""
    known = {"emotion", "style", "prosody", "sfx"}
    return [m.group(0) for m in TAG.finditer(text) if m.group(1) not in known]


def _prompt_ids(h: Higgs, text: str, n_ref: int, ref_text: str) -> list[int]:
    v = h.special
    ids = [v["<|tts|>"]]
    if ref_text and n_ref and "<|ref_text|>" in v:
        ids += [v["<|ref_text|>"], *h.tokenizer.encode(ref_text, add_special_tokens=False)]
    if n_ref:
        ids += [v["<|ref_audio|>"], *([AUDIO_PLACEHOLDER_ID] * n_ref)]
    ids += [v["<|text|>"], *h.tokenizer.encode(text, add_special_tokens=False), v["<|audio|>"]]
    return ids


# ---------------------------------------------------------------- one line

def speak(h: Higgs, text: str, refs: list[dict] | None = None, temperature: float = 0.65, top_p: float = 0.9,
          top_k: int = 30, max_new_tokens: int = 1500, seed: int = 0, label: str = "Higgs", mode: str = "graph",
          capture_step: int = 4) -> tuple[dict, dict]:
    """One line -> (ComfyUI AUDIO, info). refs = [{audio: AUDIO, text}] in order: the voice sample first,
    then the speaker's earlier lines (all in the one reference slot). mode: graph | nograph | eager (higgs_fast.py)."""
    from ..fg_audio_nodes import LineProgress
    from . import higgs_fast as fast

    t0 = time.time()
    mm.load_models_gpu([h.patcher], memory_required=h.work_bytes, force_full_load=True)
    dev = h.patcher.load_device
    m = h.module
    refs = list(refs or [])
    ref_delayed, ref_text, n_ref = None, "", 0
    if refs:
        raw = torch.cat([encode(h, r["audio"]) for r in refs], dim=0)
        ref_delayed = apply_delay_pattern(raw)
        n_ref = ref_delayed.shape[0]
        ref_text = " ".join(r.get("text", "").strip() for r in refs if r.get("text"))
    torch.manual_seed(int(seed))
    if dev.type == "xpu":
        torch.xpu.manual_seed_all(int(seed))
    elif dev.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    ids = _prompt_ids(h, text.strip(), n_ref, ref_text)
    t1 = time.time()
    budget = int(max_new_tokens) or 1500
    progress = LineProgress(text, budget, label, frames_per_s=FRAMES_PER_S)
    log.info("%s: speaking (%d prompt tokens, %d reference frames, mode %s)", label, len(ids), n_ref, mode)

    frames = []
    with torch.inference_mode():
        t = torch.tensor(ids, dtype=torch.long, device=dev)
        embeds = m.backbone.embed_tokens(t.clamp(min=0))
        mask = t == AUDIO_PLACEHOLDER_ID
        if mask.any():
            embeds[mask] = m.embedding(ref_delayed.to(dev)).to(embeds.dtype)
    if mode != "eager":
        delayed = fast.generate(h, embeds, budget, seed, {"temperature": temperature, "top_p": top_p, "top_k": top_k},
                                mode, capture_step, progress)
        frames = list(delayed)
    with torch.inference_mode():
        if mode == "eager":
            out = m.backbone(inputs_embeds=embeds.unsqueeze(0).to(h.dtype), use_cache=True)
            hidden, past, pos = out.last_hidden_state[0, -1], out.past_key_values, embeds.shape[0]
            st = SamplerState(h.num_codebooks)
            for i in range(budget):
                was_done = st.done
                codes = sampler_step(m.embedding.head(hidden.unsqueeze(0)).squeeze(0).float(), st, temperature, top_p, top_k)
                if was_done:
                    break
                frames.append(codes)
                if st.done:
                    break
                if i % 16 == 15:
                    mm.throw_exception_if_processing_interrupted()
                    progress.update(len(frames))
                nxt = m.embedding(st.last_codes.unsqueeze(0))
                out = m.backbone(inputs_embeds=nxt.unsqueeze(0).to(h.dtype), use_cache=True, past_key_values=past,
                                 cache_position=torch.tensor([pos], device=dev))
                hidden, past, pos = out.last_hidden_state[0, -1], out.past_key_values, pos + 1
            del past
            progress.done(len(frames))
        if not frames:
            raise RuntimeError("Higgs produced no audio frames")
        t2 = time.time()
        raw = reverse_delay_pattern(torch.stack(frames))
        if raw.shape[0] == 0:
            raise RuntimeError("Higgs stopped before any audio frame")
        from ..fg_audio_nodes import decode_bucketed
        cdev = next(m.codec.parameters()).device
        wav = decode_bucketed(lambda c: m.codec.decode(c.transpose(0, 1).unsqueeze(0).to(cdev)).audio_values.reshape(-1),
                              raw, 0).float().cpu()
    mm.soft_empty_cache()
    dur = wav.numel() / SAMPLE_RATE
    info = {"duration_s": round(dur, 2), "frames": int(raw.shape[0]), "prompt_tokens": len(ids), "ref_frames": int(n_ref),
            "refs": len(refs), "generate_s": round(t2 - t1, 2), "decode_s": round(time.time() - t2, 2),
            "rtf": round((time.time() - t0) / max(dur, 1e-6), 2), "frames_per_s": round(len(frames) / max(t2 - t1, 1e-6), 1),
            "prep_s": round(t1 - t0, 2), "mode": mode, "capped": len(frames) >= budget}
    if info["capped"]:
        log.warning("%s: hit the length cap (%d frames): the take probably rambled; try another seed", label, budget)
    log.info("Higgs: %s", info)
    return {"waveform": wav.view(1, 1, -1), "sample_rate": SAMPLE_RATE}, info
