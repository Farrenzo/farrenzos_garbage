"""FG Anima Role Ref KV - reference-image KV injection for Anima, as a MODEL patch.

This node does no sampling. It returns a patched MODEL that you can feed to any
ComfyUI sampler: KSampler, SamplerCustomAdvanced, FG_Advanced_KSampler, etc.

How it works
  * The Role Ref KV adapter is applied as a normal ComfyUI LoRA patch, either
    merged or bypassed. That way it composes with offloading, other LoRAs and
    quantized base weights instead of living in a private forward pass.
  * On the first conditional model call of a run, the reference latent is pushed
    once through the patched DiT at sigma 0 with the positive prompt as context.
    Its RoPE is centred on the target token grid at temporal index 10. Each
    block's self-attention K/V is cached, and the pass stops right after the
    last block's K/V is captured.
  * On every conditional call, each block's self-attention attends over
    [reference K/V ; own K/V]. Unconditional rows are left alone unless
    apply_to_uncond is on. cond and uncond can share a batch: only attention is
    split, everything else stays batched.
  * The cache is rebuilt only when the target size or positive context changes,
    and it is freed when sampling ends, so no VRAM is held between runs.
"""

import logging
import math
import os

import numpy as np
import torch
from PIL import Image

import comfy.patcher_extension
import comfy.sd
import comfy.utils
import folder_paths

try:
    from comfy.ldm.modules.attention import AttentionTensorContainer
except ImportError:  # older ComfyUI passes raw tensors to compute_attention
    AttentionTensorContainer = None

log = logging.getLogger("FG.AnimaRoleRef")

PATCH_KEY = "fg_anima_role_ref"
TARGETS = (
    "mlp.layer1",
    "mlp.layer2",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.output_proj",
)
REF_TIME_INDEX = 10.0     # reference tokens sit at a separate "frame" in the 3D RoPE
DIFFUSERS_QK_EPS = 1e-5   # Diffusers Cosmos attention; ComfyUI's Anima uses 1e-6

_ADAPTER_CACHE = {}       # path -> (stamp, state_dict); holds only the last adapter
_CAPTURES_RUNNING = [0]   # >0 while any Role Ref node is capturing; other nodes pass through


# ----------------------------------------------------------------------------- helpers

def _anima_net(model):
    net = getattr(model.model, "diffusion_model", None)
    if net is None or not hasattr(net, "llm_adapter") or not hasattr(net, "blocks"):
        raise ValueError("FG Anima Role Ref: the MODEL is not an Anima model")
    if "rope" not in str(getattr(net, "pos_emb_cls", "")).lower():
        raise ValueError("FG Anima Role Ref: this Anima build does not use RoPE positions")
    if not callable(getattr(type(net.blocks[0].self_attn), "compute_attention", None)):
        raise RuntimeError("FG Anima Role Ref: ComfyUI's Anima attention no longer exposes "
                           "compute_attention; this node needs updating for your ComfyUI version")
    return net


def _submodule(root, dotted):
    for part in dotted.split("."):
        root = getattr(root, part)
    return root


def _load_adapter(name, net):
    path = folder_paths.get_full_path_or_raise("loras", name)
    stamp = (os.path.getmtime(path), os.path.getsize(path), len(net.blocks))
    hit = _ADAPTER_CACHE.get(path)
    if hit is not None and hit[0] == stamp:
        return hit[1]

    sd = comfy.utils.load_torch_file(path, safe_load=True)
    expected = {f"diffusion_model.blocks.{b}.{t}.{s}"
                for b in range(len(net.blocks)) for t in TARGETS
                for s in ("lora_down.weight", "lora_up.weight", "alpha")}
    keys = set(sd)
    if keys != expected:
        raise ValueError(f"'{name}' is not an Anima Role Ref KV adapter for this model "
                         f"({len(expected - keys)} keys missing, {len(keys - expected)} unexpected)")
    for b, block in enumerate(net.blocks):
        for t in TARGETS:
            root = f"diffusion_model.blocks.{b}.{t}"
            down, up = sd[root + ".lora_down.weight"], sd[root + ".lora_up.weight"]
            m = _submodule(block, t)
            rank = down.shape[0]
            if tuple(down.shape) != (rank, m.in_features) or tuple(up.shape) != (m.out_features, rank):
                raise ValueError(f"Adapter shape mismatch at {root}")
            if not (torch.isfinite(down).all() and torch.isfinite(up).all()):
                raise ValueError(f"Adapter has non-finite values at {root}")

    _ADAPTER_CACHE.clear()
    _ADAPTER_CACHE[path] = (stamp, sd)
    return sd


def _encode_reference(model, vae, image, resolution):
    if image.shape[0] != 1:
        raise ValueError("FG Anima Role Ref: pass exactly one reference image")
    arr = image[0, :, :, :3].detach().float().cpu().numpy()
    pil = Image.fromarray(np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8))
    # Same preprocessing as the reference runtime: cap area at resolution^2, snap to /16.
    scale = min(1.0, math.sqrt(resolution * resolution / (pil.width * pil.height)))
    size = tuple(max(16, int(n * scale) // 16 * 16) for n in pil.size)
    if size != pil.size:
        pil = pil.resize(size, Image.Resampling.LANCZOS)
    pixels = torch.from_numpy(np.asarray(pil).copy()).float().div_(255.0).unsqueeze(0)
    latent = vae.encode(pixels)
    if latent.ndim == 4:
        latent = latent.unsqueeze(2)
    return model.model.process_latent_in(latent).float().cpu()


def _rope_table(pe, hw, center_hw, device, t_index):
    """RoPE in ComfyUI's VideoRopePosition3DEmb layout, (L, D/2, 2, 2), with the
    spatial grid centred on center_hw and an arbitrary temporal index."""
    h, w = hw

    def freqs(ntk, rng):
        return 1.0 / ((10000.0 * ntk) ** rng.to(device=device, dtype=torch.float32))

    def rot(angle):
        c, s = angle.cos(), angle.sin()
        return torch.stack([c, -s, s, c], dim=-1)

    hs = torch.arange(h, device=device, dtype=torch.float32) + (center_hw[0] // 2 - h // 2)
    ws = torch.arange(w, device=device, dtype=torch.float32) + (center_hw[1] // 2 - w // 2)
    ts = torch.tensor([t_index], device=device, dtype=torch.float32)
    et = rot(torch.outer(ts, freqs(pe.t_ntk_factor, pe.dim_temporal_range)))
    eh = rot(torch.outer(hs, freqs(pe.h_ntk_factor, pe.dim_spatial_range)))
    ew = rot(torch.outer(ws, freqs(pe.w_ntk_factor, pe.dim_spatial_range)))
    em = torch.cat([
        et[:, None, None].expand(1, h, w, -1, -1),
        eh[None, :, None].expand(1, h, w, -1, -1),
        ew[None, None, :].expand(1, h, w, -1, -1),
    ], dim=-2)
    return em.reshape(h * w, em.shape[-2], 2, 2).float()


def _peek(t):
    return t.peek() if AttentionTensorContainer is not None and isinstance(t, AttentionTensorContainer) else t


def _take(t):
    return t.take() if AttentionTensorContainer is not None and isinstance(t, AttentionTensorContainer) else t


class _CaptureComplete(Exception):
    pass


# ----------------------------------------------------------------------------- runtime state

class _RoleRefState:
    def __init__(self, ref_latent, n_blocks, start_percent, end_percent, apply_to_uncond, sigma_range):
        self.ref_latent = ref_latent
        self.n_blocks = n_blocks
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.apply_to_uncond = apply_to_uncond
        self.sigma_range = sigma_range    # (high, low), refreshed in on_pre_run
        self.rope_checked = False
        self._reset()

    def _reset(self):
        self.cache = None                 # list[(k, v)] per block, each (1, L_ref, H, D)
        self.cache_key = None
        self.cache_ctx = None
        self.active = None                # (cache, cond_idx, uncond_idx) during a forward
        self.capturing = None             # dict while the reference pass runs

    # -- callbacks
    def on_pre_run(self, patcher):
        self._reset()
        ms = patcher.get_model_object("model_sampling")
        self.sigma_range = (float(ms.percent_to_sigma(self.start_percent)),
                            float(ms.percent_to_sigma(self.end_percent)))

    def on_cleanup(self, patcher):
        self._reset()

    # -- diffusion model wrapper
    def diffusion_wrapper(self, executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        if self.capturing is None:
            to = kwargs.get("transformer_options", {})
            plan = self._plan(x.shape[0], to)
            if plan is not None:
                flags, cou = plan
                cache = self._ensure_cache(executor.class_obj, x, timesteps, context, cou, to)
                if cache is not None:
                    if all(flags):
                        self.active = (cache, None, None)
                    else:
                        f = torch.tensor(flags, device=x.device)
                        self.active = (cache, f.nonzero().flatten(), (~f).nonzero().flatten())
                    try:
                        return executor(x, timesteps, context, fps, padding_mask, **kwargs)
                    finally:
                        self.active = None
        return executor(x, timesteps, context, fps, padding_mask, **kwargs)

    def _plan(self, batch, to):
        sigmas = to.get("sigmas")
        if sigmas is not None and self.sigma_range is not None:
            s = float(sigmas.max())
            high, low = self.sigma_range
            if s > high * (1 + 1e-5) + 1e-8 or s < low * (1 - 1e-5) - 1e-8:
                return None
        cou = to.get("cond_or_uncond")
        if not cou or batch % len(cou):
            return None
        per = batch // len(cou)
        flags = [c == 0 or self.apply_to_uncond for c in cou for _ in range(per)]
        return (flags, cou) if any(flags) else None

    def _ensure_cache(self, net, x, timesteps, context, cou, to):
        key = (tuple(x.shape[-2:]), x.dtype, x.device)
        per = x.shape[0] // len(cou)
        pos_rows = [j * per for j, c in enumerate(cou) if c == 0]
        if not pos_rows:  # uncond-only call (apply_to_uncond): reuse the positive cache
            return self.cache if self.cache_key == key else None
        ctx = context[pos_rows[0]:pos_rows[0] + 1]
        if (self.cache is not None and self.cache_key == key
                and self.cache_ctx.shape == ctx.shape and torch.equal(self.cache_ctx, ctx)):
            return self.cache
        self.cache = None
        self.cache = self._capture(net, x, timesteps, ctx, to)
        self.cache_key, self.cache_ctx = key, ctx.clone()
        return self.cache

    def _capture(self, net, x, timesteps, ctx, to):
        ps = net.patch_spatial
        ref = self.ref_latent.to(device=x.device, dtype=x.dtype)
        ref_hw = (ref.shape[-2] // ps, ref.shape[-1] // ps)
        tgt_hw = (math.ceil(x.shape[-2] / ps), math.ceil(x.shape[-1] / ps))

        if not self.rope_checked:
            native = net.pos_embedder.generate_embeddings((1, 1, *tgt_hw, 0), fps=None, device=x.device)
            mine = _rope_table(net.pos_embedder, tgt_hw, tgt_hw, x.device, 0.0)
            if native.shape != mine.shape or not torch.allclose(native.float(), mine, atol=1e-5):
                raise RuntimeError("FG Anima Role Ref: ComfyUI's Anima RoPE layout changed; "
                                   "this node needs updating for your ComfyUI version")
            self.rope_checked = True

        rope = _rope_table(net.pos_embedder, ref_hw, tgt_hw, x.device, REF_TIME_INDEX)[None, :, None]

        def rope_override(q, k, v, pe=None, attn_mask=None, extra_options=None):
            return {"pe": rope}

        cto = dict(to)
        patches = dict(cto.get("patches", {}))
        patches["attn1_patch"] = list(patches.get("attn1_patch", [])) + [rope_override]
        cto["patches"] = patches

        t0 = torch.zeros((1,), device=x.device, dtype=timesteps.dtype)
        self.capturing = {}
        _CAPTURES_RUNNING[0] += 1
        try:
            # _forward skips the DIFFUSION_MODEL wrapper chain, so caching wrappers
            # (TeaCache and the like) never see the reference pass.
            net._forward(ref, t0, ctx, None, None, transformer_options=cto)
        except _CaptureComplete:
            pass
        finally:
            _CAPTURES_RUNNING[0] -= 1
            kv, self.capturing = self.capturing, None
        if len(kv) != self.n_blocks:
            raise RuntimeError(f"FG Anima Role Ref: captured {len(kv)}/{self.n_blocks} blocks")
        return [kv[i] for i in range(self.n_blocks)]


def _attention_hook(state, index, prev):
    def compute_attention(q, k, v, transformer_options={}):
        if state.capturing is not None:
            state.capturing[index] = (_peek(k).clone(), _peek(v).clone())
            if index == state.n_blocks - 1:
                raise _CaptureComplete()
            return prev(q, k, v, transformer_options=transformer_options)

        active = state.active
        if active is None or _CAPTURES_RUNNING[0]:  # never inject into another node's reference pass
            return prev(q, k, v, transformer_options=transformer_options)

        cache, cond_idx, uncond_idx = active
        rk, rv = cache[index]
        wrap = AttentionTensorContainer if isinstance(q, AttentionTensorContainer or ()) else (lambda t: t)
        q, k, v = _take(q), _take(k), _take(v)

        def with_ref(qs, ks, vs):
            n = ks.shape[0]
            ks = torch.cat([rk.to(ks.dtype).expand(n, -1, -1, -1), ks], dim=1)
            vs = torch.cat([rv.to(vs.dtype).expand(n, -1, -1, -1), vs], dim=1)
            return prev(wrap(qs), wrap(ks), wrap(vs), transformer_options=transformer_options)

        if cond_idx is None:
            return with_ref(q, k, v)
        out_c = with_ref(q[cond_idx], k[cond_idx], v[cond_idx])
        out_u = prev(wrap(q[uncond_idx]), wrap(k[uncond_idx]), wrap(v[uncond_idx]),
                     transformer_options=transformer_options)
        out = out_c.new_empty((q.shape[0],) + tuple(out_c.shape[1:]))
        out[cond_idx] = out_c
        out[uncond_idx] = out_u
        return out

    return compute_attention


# ----------------------------------------------------------------------------- node

class FG_AnimaRoleRefKV:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "vae": ("VAE",),
            "reference": ("IMAGE",),
            "adapter": (["None"] + folder_paths.get_filename_list("loras"), {
                "tooltip": "Anima Role Ref KV adapter. 'None' if you load it yourself with a LoRA loader."}),
            "adapter_strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01}),
            "adapter_mode": (["merge", "bypass"], {
                "default": "merge",
                "tooltip": "merge: baked into the weights (fastest). bypass: unmerged side path, "
                           "closer to the reference runtime and works on quantized weights."}),
            "reference_resolution": ("INT", {
                "default": 768, "min": 256, "max": 2048, "step": 16,
                "tooltip": "Reference is downscaled to at most this^2 pixels (snapped to /16)."}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            "apply_to_uncond": ("BOOLEAN", {
                "default": False,
                "tooltip": "Also inject the reference into the negative pass. Off matches the "
                           "reference runtime; CFG then amplifies the reference."}),
            "diffusers_qk_eps": ("BOOLEAN", {
                "default": True,
                "tooltip": "Use eps=1e-5 on the DiT q/k norms like the Diffusers runtime the "
                           "adapter targets (ComfyUI default is 1e-6)."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Farrenzo's Garbage/Anima"
    DESCRIPTION = "Patches an Anima MODEL with Role Ref KV reference injection. Sample it with any sampler."

    def apply(self, model, vae, reference, adapter, adapter_strength, adapter_mode,
              reference_resolution, start_percent, end_percent, apply_to_uncond, diffusers_qk_eps):
        if start_percent > end_percent:
            raise ValueError("start_percent must be <= end_percent")
        net = _anima_net(model)
        n_blocks = len(net.blocks)
        ref_latent = _encode_reference(model, vae, reference, reference_resolution)

        if adapter != "None" and adapter_strength != 0:
            sd = _load_adapter(adapter, net)
            loader = (comfy.sd.load_bypass_lora_for_models if adapter_mode == "bypass"
                      else comfy.sd.load_lora_for_models)
            patched, _ = loader(model, None, sd, adapter_strength, 0)
        else:
            patched = model.clone()

        ms = patched.get_model_object("model_sampling")
        state = _RoleRefState(ref_latent, n_blocks, start_percent, end_percent, apply_to_uncond,
                              (float(ms.percent_to_sigma(start_percent)), float(ms.percent_to_sigma(end_percent))))

        for i, block in enumerate(net.blocks):
            key = f"diffusion_model.blocks.{i}.self_attn.compute_attention"
            if key in patched.object_patches:      # chain onto an upstream patch in this graph
                prev = patched.object_patches[key]
            else:                                  # class method: ignore stale instance leftovers
                prev = type(block.self_attn).compute_attention.__get__(block.self_attn)
            patched.add_object_patch(key, _attention_hook(state, i, prev))
            if diffusers_qk_eps:
                for attn in ("self_attn", "cross_attn"):
                    for norm in ("q_norm", "k_norm"):
                        patched.add_object_patch(f"diffusion_model.blocks.{i}.{attn}.{norm}.eps", DIFFUSERS_QK_EPS)

        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, PATCH_KEY,
                                     state.diffusion_wrapper)
        patched.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_PRE_RUN, PATCH_KEY, state.on_pre_run)
        patched.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_CLEANUP, PATCH_KEY, state.on_cleanup)
        log.info("Role Ref KV: reference latent %s, adapter=%s (%s, %.2f)",
                 tuple(ref_latent.shape), adapter, adapter_mode, adapter_strength)
        return (patched,)

