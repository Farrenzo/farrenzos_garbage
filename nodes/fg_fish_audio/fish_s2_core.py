"""Fish Audio S2 Pro (fishaudio/s2-pro) inside ComfyUI, on XPU / CUDA.

After Renzo Studio's .planning/audio_plan/run_fish_s2_xpu.py. The fish-speech package sits next to
this file (never pip-installed) and descript_shim.py stands in for the `dac` / `audiotools`
packages its codec imports. fish-speech's own model and sampler run unchanged; this file adds:

* a streaming loader: a meta-device skeleton filled in one pass over the shards (the stock loader
  holds every shard in host RAM first, ~19 GB peak);
* the model + codec wrapped in a ModelPatcher, so ComfyUI moves it on / off the GPU like any other
  model (an H3 or SCAIL render can take the memory back);
* one script line per conversation turn: the system message holds every speaker's reference
  (`<|speaker:i|>` + transcript, then their codes), then the earlier turns of the scene (text + the
  audio actually generated for them), then the new line. The oldest turns are dropped until the
  prompt fits in `max_seq_len - 2048` (fish-speech's limit; 2048 is room for the new line);
* ComfyUI's progress bar and Cancel inside the token loop;
* a faster decode loop with XPU graph capture (fish_s2_fast.py; mode "stock" keeps fish-speech's own).

License: fish-speech and the S2 Pro weights are under the Fish Audio Research License
(fish_speech/LICENSE): research and non-commercial use.
"""
import contextlib
import hashlib
import json
import logging
import os
import sys
import time
import types
from pathlib import Path

import torch

import comfy.model_management as mm
import comfy.model_patcher

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # fish-speech imports itself as `fish_speech`

from . import descript_shim  # noqa: E402

descript_shim.install()
if "loralib" not in sys.modules:
    # llama.py imports its LoRA helper, which imports loralib at the top; only fine-tuning calls it
    sys.modules["loralib"] = types.ModuleType("loralib")

from loguru import logger as _loguru  # noqa: E402

_loguru.disable("fish_speech")  # fish-speech logs every prompt at INFO; keep ComfyUI's console readable

log = logging.getLogger("FG_FishS2")
SAMPLE_RATE = 44100
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# ---------------------------------------------------------------- loading

class FishS2Module(torch.nn.Module):
    """Model + codec in one module, so one ModelPatcher moves both."""

    def __init__(self, lm, codec):
        super().__init__()
        self.lm = lm
        self.codec = codec


class FishS2:
    """What the loader node outputs: the patcher plus what speaking needs."""

    def __init__(self, patcher, cfg, path: str, work_bytes: int):
        self.patcher = patcher
        self.cfg = cfg
        self.path = path
        self.work_bytes = work_bytes
        self.codes_cache: dict[str, torch.Tensor] = {}
        self.decoder = None  # fish_s2_fast.Decoder, built on first use on the load device

    @property
    def module(self) -> FishS2Module:
        return self.patcher.model


def _build_codec(codec_path: Path):
    """The DAC codec built directly (fish_speech/configs/modded_dac_vq.yaml, one for one), no hydra."""
    from functools import partial

    from fish_speech.models.dac.modded_dac import DAC, ModelArgs, WindowLimitedTransformer
    from fish_speech.models.dac.rvq import DownsampleResidualVectorQuantize

    def transformer():
        return WindowLimitedTransformer(
            config=ModelArgs(block_size=2048, n_layer=8, n_head=16, dim=1024, intermediate_size=3072, n_local_heads=-1,
                             head_dim=64, rope_base=10000, norm_eps=1e-5, dropout_rate=0.1, attn_dropout_rate=0.1,
                             channels_first=True),
            input_dim=1024, window_size=128, causal=True)

    general = partial(ModelArgs, block_size=8192, n_local_heads=-1, head_dim=64, rope_base=10000, norm_eps=1e-5,
                      dropout_rate=0.1, attn_dropout_rate=0.1, channels_first=True)
    quantizer = DownsampleResidualVectorQuantize(input_dim=1024, n_codebooks=9, codebook_size=1024, codebook_dim=8,
                                                 quantizer_dropout=0.5, downsample_factor=[2, 2], post_module=transformer(),
                                                 pre_module=transformer(), semantic_codebook_size=4096)
    codec = DAC(sample_rate=SAMPLE_RATE, encoder_dim=64, encoder_rates=[2, 4, 8, 8], decoder_dim=1536,
                decoder_rates=[8, 8, 4, 2], encoder_transformer_layers=[0, 0, 0, 4], decoder_transformer_layers=[4, 0, 0, 0],
                transformer_general_config=general, quantizer=quantizer)
    sd = torch.load(str(codec_path), map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    if any("generator" in k for k in sd):
        sd = {k.replace("generator.", ""): v for k, v in sd.items() if "generator." in k}
    res = codec.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if "kv_cache" not in k]
    if missing:
        raise RuntimeError(f"Fish S2 codec: {len(missing)} weights missing, e.g. {missing[:5]}")
    return codec.eval()


def load(path: str, max_seq_len: int = 8192, precision: str = "bfloat16", codec_precision: str = "float32") -> FishS2:
    """Model + codec on the CPU (ComfyUI moves them to the GPU when a line is spoken)."""
    from fish_speech.models.text2semantic import llama
    from fish_speech.models.text2semantic.llama import (BaseModelArgs, DualARTransformer,
                                                        _remap_fish_qwen3_omni_keys, precompute_freqs_cis)
    from fish_speech.tokenizer import FishTokenizer
    from safetensors import safe_open

    t0 = time.time()
    ckpt = Path(path)
    for need in ("config.json", "codec.pth"):
        if not (ckpt / need).is_file():
            raise FileNotFoundError(f"Fish S2: {need} not found in {ckpt}")
    dtype = DTYPES[precision]
    if mm.get_torch_device().type == "xpu":
        # llama.py pins FLASH_ATTENTION on its no-mask branch, which raises on XPU; let SDPA choose
        llama.sdpa_kernel = lambda *_a, **_k: contextlib.nullcontext()

    cfg = BaseModelArgs.from_pretrained(str(ckpt))
    cfg.max_seq_len = int(max_seq_len)
    tok = FishTokenizer.from_pretrained(str(ckpt))
    cfg.semantic_begin_id, cfg.semantic_end_id = tok.semantic_begin_id, tok.semantic_end_id
    with torch.device("meta"):
        lm = DualARTransformer(cfg)
    lm.tokenizer = tok

    index = ckpt / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file() else ["model.safetensors"]
    state = {}
    for shard in shards:
        with safe_open(str(ckpt / shard), framework="pt") as f:
            for key in f.keys():
                name = next(iter(_remap_fish_qwen3_omni_keys({key: None})))
                t = f.get_tensor(key)
                state[name] = t.to(dtype) if t.is_floating_point() else t
                del t
    missing, unexpected = lm.load_state_dict(state, strict=False, assign=True)
    state.clear()
    missing = [m for m in missing if not m.endswith(("freqs_cis", "causal_mask"))]
    if missing:
        raise RuntimeError(f"Fish S2: weights missing, e.g. {missing[:6]}")
    if unexpected:
        log.info("Fish S2: %d checkpoint tensors not used, e.g. %s", len(unexpected), unexpected[:4])

    # non-persistent buffers stayed on meta
    lm.freqs_cis = precompute_freqs_cis(cfg.max_seq_len, cfg.head_dim, cfg.rope_base).to(dtype)
    lm.fast_freqs_cis = precompute_freqs_cis(cfg.num_codebooks, cfg.fast_head_dim, cfg.rope_base).to(dtype)
    lm.causal_mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
    lm.setup_caches(max_batch_size=1, max_seq_len=cfg.max_seq_len, dtype=dtype)
    lm._cache_setup_done = True
    left = [n for n, t in list(lm.named_parameters()) + list(lm.named_buffers()) if t.is_meta]
    if left:
        raise RuntimeError(f"Fish S2: {len(left)} tensors still on meta, e.g. {left[:4]}")

    codec = _build_codec(ckpt / "codec.pth").to(DTYPES[codec_precision])
    module = FishS2Module(lm, codec).eval()
    for p in module.parameters():
        p.requires_grad_(False)
    patcher = comfy.model_patcher.CoreModelPatcher(module, load_device=mm.get_torch_device(),
                                                   offload_device=mm.unet_offload_device())
    # working memory beyond the weights: decode activations of the codec for a long line, sampling buffers
    work = 1536 * 2**20
    log.info("Fish S2 loaded from %s in %.1f s (max_seq_len %d)", ckpt, time.time() - t0, cfg.max_seq_len)
    return FishS2(patcher, cfg, str(ckpt), work)


# ---------------------------------------------------------------- audio <-> codes

def _mono(audio: dict, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """ComfyUI AUDIO {waveform [B, C, T], sample_rate} -> mono float32 [T] at the codec rate."""
    wav = audio["waveform"][0].float()
    wav = wav.mean(dim=0) if wav.dim() == 2 else wav
    sr = int(audio["sample_rate"])
    if sr != sample_rate:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


@torch.inference_mode()
def encode(fish: FishS2, audio: dict) -> torch.Tensor:
    """AUDIO -> codes [num_codebooks, T] (CPU), cached by content."""
    wav = _mono(audio)
    key = hashlib.sha1(wav.numpy().tobytes()).hexdigest()
    if key in fish.codes_cache:
        return fish.codes_cache[key]
    codec = fish.module.codec
    p = next(codec.parameters())
    idx, lens = codec.encode(wav[None, None].to(p.device, p.dtype), torch.tensor([wav.numel()], device=p.device))
    codes = idx[0, :, : lens[0]].cpu()
    fish.codes_cache[key] = codes
    return codes


# ---------------------------------------------------------------- one line

def _progress_loop(fish_inference, progress):
    """Swap fish-speech's per-token tqdm (stock mode) for the line's progress + Cancel."""
    def loop(iterable, **_kw):
        n = 0
        try:
            for n, x in enumerate(iterable, start=1):
                mm.throw_exception_if_processing_interrupted()
                yield x
                if n % 8 == 0:
                    progress.update(n)
        finally:
            progress.done(n)
    fish_inference.tqdm = loop


def _conversation(fish: FishS2, voices: list[dict], history: list[dict], line: str):
    from fish_speech.content_sequence import TextPart, VQPart
    from fish_speech.conversation import Conversation, Message

    conv = Conversation()
    if voices:
        texts = [f"<|speaker:{i}|>{v['text']}" for i, v in enumerate(voices)]
        codes = torch.cat([encode(fish, v["audio"]) for v in voices], dim=1)
        parts = [TextPart(text="convert the provided text to speech reference to the following:\n\nText:\n", cal_loss=False),
                 TextPart(text="\n".join(texts), cal_loss=False), TextPart(text="\n\nSpeech:\n", cal_loss=False),
                 VQPart(codes=codes, cal_loss=False)]
    else:
        parts = [TextPart(text="convert the provided text to speech", cal_loss=False)]
    conv.append(Message(role="system", parts=parts, cal_loss=False, add_im_start=True, add_im_end=True))
    for h in history:
        conv.append(Message(role="user", parts=[TextPart(text=f"<|speaker:{h['speaker']}|>{h['text']}", cal_loss=False)],
                            cal_loss=False, add_im_start=True, add_im_end=True))
        conv.append(Message(role="assistant", parts=[VQPart(codes=encode(fish, h["audio"]), cal_loss=False)],
                            cal_loss=False, modality="voice", add_im_start=True, add_im_end=True))
    conv.append(Message(role="user", parts=[TextPart(text=line, cal_loss=False)], cal_loss=False, add_im_start=True, add_im_end=True))
    conv.append(Message(role="assistant", parts=[], cal_loss=False, modality="voice", add_im_start=True, add_im_end=False))
    return conv.encode_for_inference(fish.module.lm.tokenizer, num_codebooks=fish.cfg.num_codebooks)


def speak(fish: FishS2, text: str, speaker: int = 0, voices: list[dict] | None = None, history: list[dict] | None = None,
          temperature: float = 0.8, top_p: float = 0.9, top_k: int = 30, max_new_tokens: int = 0, seed: int = 0,
          mode: str = "graph", capture_step: int = 4, label: str = "Fish S2") -> tuple[dict, dict]:
    """One line as one turn -> (ComfyUI AUDIO, info). mode: graph | nograph | stock (fish_s2_fast.py)."""
    from fish_speech.models.text2semantic import inference

    from . import fish_s2_fast as fast

    t0 = time.time()
    mm.load_models_gpu([fish.patcher], memory_required=fish.work_bytes, force_full_load=True)
    dev = fish.patcher.load_device
    lm = fish.module.lm
    fast.set_attention(mode != "stock")

    voices = list(voices or [])
    history = list(history or [])
    line = text.strip()
    if not line.startswith("<|speaker:"):
        line = f"<|speaker:{int(speaker)}|>{line}"
    limit = fish.cfg.max_seq_len - 2048
    while True:
        encoded, audio_masks, audio_parts = _conversation(fish, voices, history, line)
        if encoded.size(1) <= limit or not history:
            break
        history.pop(0)  # sliding window: the oldest turn goes first
    if encoded.size(1) > limit:
        raise ValueError(f"Fish S2: the voice references and this line take {encoded.size(1)} tokens; max_seq_len "
                         f"{fish.cfg.max_seq_len} leaves {limit}. Use shorter references or raise max_seq_len on the loader.")

    torch.manual_seed(seed)
    if dev.type == "xpu":
        torch.xpu.manual_seed_all(seed)
    elif dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    t1 = time.time()
    prompt_len = encoded.size(1)
    budget = min(int(max_new_tokens), fish.cfg.max_seq_len - prompt_len) if max_new_tokens else fish.cfg.max_seq_len - prompt_len
    from ..fg_audio_nodes import LineProgress
    progress = LineProgress(line, budget, label, frames_per_s=21.5)
    log.info("%s: speaking (speaker %s, %d prompt tokens, mode %s)", label, speaker, prompt_len, mode)
    if mode == "stock":
        _progress_loop(inference, progress)
        y = inference.generate(model=lm, prompt=encoded.to(dev), max_new_tokens=int(max_new_tokens), audio_masks=audio_masks,
                               audio_parts=audio_parts, decode_one_token=inference.decode_one_token_ar,
                               temperature=float(temperature), top_p=float(top_p), top_k=int(top_k))
        codes = y[1:, prompt_len:-1].clone()
        del y
    else:
        if fish.decoder is None or fish.decoder.dev != dev or fish.decoder.lm is not lm:
            fish.decoder = fast.Decoder(lm, dev)
        codes = fast.generate(fish.decoder, encoded.to(dev), audio_masks, audio_parts, max_new_tokens, seed,
                              {"temperature": temperature, "top_p": top_p, "top_k": top_k}, mode, capture_step, progress)
    t2 = time.time()
    if codes.shape[1] == 0:
        raise RuntimeError("Fish S2 produced no audio frames")
    codec = fish.module.codec
    with torch.inference_mode():
        from ..fg_audio_nodes import decode_bucketed
        cdev = next(codec.parameters()).device
        wav = decode_bucketed(lambda c: codec.from_indices(c[None].to(cdev))[0, 0], codes, 1).float().cpu()
    mm.soft_empty_cache()
    dur = wav.numel() / SAMPLE_RATE
    info = {"duration_s": round(dur, 2), "frames": int(codes.shape[1]), "prompt_tokens": int(prompt_len),
            "history_used": len(history), "voices": len(voices), "mode": mode, "generate_s": round(t2 - t1, 2),
            "decode_s": round(time.time() - t2, 2), "rtf": round((time.time() - t0) / max(dur, 1e-6), 2),
            "frames_per_s": round(codes.shape[1] / max(t2 - t1, 1e-6), 1)}
    log.info("Fish S2: %s", info)
    return {"waveform": wav.view(1, 1, -1), "sample_rate": SAMPLE_RATE}, info
