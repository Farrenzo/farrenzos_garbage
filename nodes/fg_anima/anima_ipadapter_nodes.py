"""ComfyUI custom nodes for Anima IP-Adapter.

Two checkpoint families are supported:

* **v3 / TimeResampler** (``resampler.*`` + ``ip_cross_attns.*``) — the
  InstantCharacter-style adapter built on :class:`IPAdapterSigLIP`.
* **KV / LuciferTC** (``blocks.N.ip_k_proj`` + ``blocks.N.adaln_ip`` +
  ``lora.base_model.model.*``) — classic decoupled cross-attention with **no**
  resampler, shipped together with a co-trained rank-32 LoRA over the whole DiT.
  The LoRA is not optional: the IP branch was trained with it active.

Injection is scoped to a DIFFUSION_MODEL wrapper, so cross_attn.forward is
patched immediately before the model runs and restored in a ``finally``. That
keeps the hook from leaking into other workflows sharing the same checkpoint,
and hands us the current timestep for free.

!! READ BEFORE TRUSTING OUTPUT !!
The KV branch is reconstructed from tensor shapes, not from the author's
reference implementation. Two details cannot be recovered from a state dict and
are exposed as node inputs instead of guessed:

    * `inject_point` - whether the IP result is summed before or after the
                       block's shared ``output_proj``.
    * `adaln_source` - what conditioning vector drives ``adaln_ip``.

Sweep them once against a known-good reference image and lock in what looks
right, or better, read the values off the training repo.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.lora
import comfy.model_management as mm
import comfy.patcher_extension
import folder_paths
from PIL import Image as PILImage

from .anima_methods import IPAdapterSigLIP

IP_WRAPPER_KEY = "anima_ip_adapter"


class AnyType(str):
    """Type that compares equal to every other type, for a pure-ordering pin."""

    def __ne__(self, other):
        return False


ANY = AnyType("*")

_SIGLIP_CACHE = {"model": None, "proc": None}


# ─── SigLIP2 loading ──────────────────────────────────────────────────

def _siglip2_dir() -> str:
    # Deferred to avoid a circular import at module load.
    from ... import GOOGLE_SIGLIP2_DIR, GOOGLE_SIGLIP2_INFO

    if not os.path.isdir(GOOGLE_SIGLIP2_DIR) or not os.listdir(GOOGLE_SIGLIP2_DIR):
        raise FileNotFoundError(
            f"SigLIP2 not found at {GOOGLE_SIGLIP2_DIR}. Download "
            f"{GOOGLE_SIGLIP2_INFO['hf_repo']} into that folder, e.g.:\n"
            f"  huggingface-cli download {GOOGLE_SIGLIP2_INFO['hf_repo']} "
            f"--local-dir \"{GOOGLE_SIGLIP2_DIR}\""
        )
    return GOOGLE_SIGLIP2_DIR


def _load_siglip2():
    if _SIGLIP_CACHE["model"] is not None:
        return _SIGLIP_CACHE["model"], _SIGLIP_CACHE["proc"]

    from transformers import AutoImageProcessor, SiglipVisionModel

    path = _siglip2_dir()
    device = mm.text_encoder_device()
    dtype = mm.text_encoder_dtype(device)

    model = SiglipVisionModel.from_pretrained(
        path, torch_dtype=dtype, local_files_only=True,
    ).to(device).eval()
    model.requires_grad_(False)

    # transformers renamed use_fast -> backend; keep both paths working.
    try:
        proc = AutoImageProcessor.from_pretrained(
            path, local_files_only=True, backend="torchvision",
        )
    except TypeError:
        proc = AutoImageProcessor.from_pretrained(
            path, local_files_only=True, use_fast=True,
        )

    _SIGLIP_CACHE["model"], _SIGLIP_CACHE["proc"] = model, proc
    return model, proc


def prepare_siglip_image(pil, size=512, pad=(255, 255, 255)):
    """Letterbox to a square on a white field, then resize. SigLIP2 is a fixed
    512x512 patch16 tower, so every input becomes 32x32 = 1024 patches no matter
    its aspect ratio. Used by BOTH the encoder and the visualiser — if these two
    ever diverge, the heatmap stops describing what the model actually saw."""
    w, h = pil.size
    if w == h:
        return pil.resize((size, size), PILImage.BICUBIC)
    side = max(w, h)
    canvas = PILImage.new("RGB", (side, side), pad)
    canvas.paste(pil, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), PILImage.BICUBIC)


def purge_siglip2():
    """Called by FG_PurgeVRAM."""
    if _SIGLIP_CACHE["model"] is not None:
        _SIGLIP_CACHE["model"].to("cpu")
    _SIGLIP_CACHE["model"] = None
    _SIGLIP_CACHE["proc"] = None
    mm.soft_empty_cache()


# ─── KV-format adapter (LuciferTC) ────────────────────────────────────

class AnimaIPBlockKV(nn.Module):
    """One block's decoupled K/V projections plus the adaLN gate.

    Parameter names are chosen so the checkpoint loads cleanly:
    ``blocks.{i}.ip_k_proj.weight`` / ``blocks.{i}.adaln_ip.1.weight``.
    """

    def __init__(self, dit_dim: int, img_dim: int, has_adaln: bool, adaln_in: int):
        super().__init__()
        self.ip_k_proj = nn.Linear(img_dim, dit_dim)
        self.ip_v_proj = nn.Linear(img_dim, dit_dim)
        self.adaln_ip = (
            nn.Sequential(nn.SiLU(), nn.Linear(adaln_in, dit_dim)) if has_adaln else None
        )


class AnimaIPAdapterKV(nn.Module):
    """Decoupled cross-attention adapter: reuses the host block's query."""

    kind = "kv"

    def __init__(self, num_blocks: int, dit_dim: int, img_dim: int,
                 has_adaln: bool, adaln_in: int):
        super().__init__()
        self.num_blocks = num_blocks
        self.dit_dim = dit_dim
        self.img_dim = img_dim
        self.blocks = nn.ModuleList([
            AnimaIPBlockKV(dit_dim, img_dim, has_adaln, adaln_in)
            for _ in range(num_blocks)
        ])

    def forward_block(self, bidx, x, host_block, image_tokens, cond_vec,
                      inject_point, reuse_k_norm=True, capture=None):
        """x: (B, S, D) the block's normalized input. image_tokens: (B, N, img_dim).

        Mirrors comfy.ldm.cosmos.predict2.Attention.compute_qkv: project, split
        into (B, S, H, D), then RMSNorm over the head dim. RoPE is skipped
        because cross-attention never applies it (``is_selfattn`` is False).
        """
        ip = self.blocks[bidx]
        B, S, _ = x.shape

        heads = getattr(host_block, "n_heads", None) or getattr(host_block, "num_heads", None)
        head_dim = getattr(host_block, "head_dim", None)
        if heads is None or head_dim is None:
            raise RuntimeError(
                "cross_attn exposes neither n_heads nor head_dim; cannot split "
                "the IP branch into heads."
            )
        inner = heads * head_dim
        if inner != self.dit_dim:
            raise RuntimeError(
                f"IP-Adapter was trained for inner dim {self.dit_dim} but this "
                f"block's attention inner dim is {inner} "
                f"({heads} heads x {head_dim}). Wrong Anima variant."
            )

        q = host_block.q_proj(x).view(B, S, heads, head_dim)
        q = host_block.q_norm(q)

        img = image_tokens.to(dtype=q.dtype, device=q.device)
        if img.shape[0] != B:
            img = (img.expand(B, -1, -1) if img.shape[0] == 1
                   else img.repeat(B // img.shape[0], 1, 1))
        N = img.shape[1]

        k = ip.ip_k_proj(img).view(B, N, heads, head_dim)
        v = ip.ip_v_proj(img).view(B, N, heads, head_dim)
        if reuse_k_norm and hasattr(host_block, "k_norm"):
            k = host_block.k_norm(k)
        if hasattr(host_block, "v_norm"):
            v = host_block.v_norm(v)

        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if capture is not None:
            # Real attention mass per image patch: softmax(qk^T) summed over
            # every query position. Chunked over S so the S x N matrix never
            # materialises in full.
            with torch.no_grad():
                scale = 1.0 / math.sqrt(head_dim)
                acc = torch.zeros(N, device=qh.device, dtype=torch.float32)
                for i in range(0, S, 1024):
                    w = (qh[:, :, i:i + 1024].float() @ kh.float().transpose(-1, -2)) * scale
                    acc += w.softmax(dim=-1).sum(dim=(0, 1, 2))
                capture.append(acc / max(1, S * B * heads))

        out = F.scaled_dot_product_attention(qh, kh, vh)
        out = out.transpose(1, 2).reshape(B, S, inner)

        if ip.adaln_ip is not None and cond_vec is not None:
            gate = ip.adaln_ip(cond_vec.to(dtype=out.dtype, device=out.device))
            if gate.dim() == 2:
                gate = gate.unsqueeze(1)
            out = out * (1.0 + gate)

        if inject_point == "shared_output_proj":
            # output_proj is Linear(bias=False), so projecting the IP result here
            # is exactly equivalent to summing the two attention results before
            # the shared projection.
            out = host_block.output_proj(out)

        return out


def _build_lora_patches(state, alpha: float = 32.0):
    """``lora.base_model.model.<target>.lora_{A,B}.weight`` -> comfy patch dict.

    Built through comfy.lora.load_lora so the patches come back as whatever
    weight_adapter type this ComfyUI build expects. Hand-rolled ("lora", (...))
    tuples are no longer recognised by calculate_weight.
    """
    prefix = "lora.base_model.model."
    lora_sd, targets = {}, set()
    for k, v in state.items():
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix):]
        lora_sd[rest] = v
        for suffix in (".lora_A.weight", ".lora_B.weight"):
            if rest.endswith(suffix):
                targets.add(rest[: -len(suffix)])

    if not targets:
        return {}

    # PEFT stores alpha in adapter_config.json, which is not shipped inside the
    # safetensors. Inject it so LoRAAdapter scales by alpha/rank explicitly.
    for t in targets:
        lora_sd[f"{t}.alpha"] = torch.tensor(float(alpha))

    to_load = {t: f"diffusion_model.{t}.weight" for t in targets}
    return comfy.lora.load_lora(lora_sd, to_load, log_missing=False)


# ─── Injection hook ───────────────────────────────────────────────────

class IPAdapterHook:
    """Patches each DiT block's cross_attn.forward for the duration of one pass."""

    def __init__(self, ip_adapter, weight, sigma_start, sigma_end,
                 inject_point="direct", adaln_source="off", reuse_k_norm=True,
                 capture=False):
        self.ip = ip_adapter
        self.weight = weight
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.inject_point = inject_point
        self.adaln_source = adaln_source
        self.reuse_k_norm = reuse_k_norm
        self.capture = capture

        self.image_tokens = None
        self.siglip_features = None
        self._sigma = None
        self._patches = []

    # -- per-pass state ------------------------------------------------

    def set_timestep(self, timestep):
        try:
            self._sigma = float(torch.as_tensor(timestep).max().item())
        except Exception:
            self._sigma = None

        if getattr(self.ip, "kind", None) != "kv" and self.siglip_features is not None:
            # TimeResampler is timestep-conditioned; re-encode every step.
            with torch.no_grad():
                ts = torch.as_tensor(timestep).flatten()
                self.image_tokens = self.ip.encode_ref(self.siglip_features, timestep=ts)

    def _active(self) -> bool:
        if self.weight <= 0 or self.image_tokens is None:
            return False
        if self._sigma is None:
            return True
        return self.sigma_end <= self._sigma <= self.sigma_start

    def _cond_vec(self, x):
        if self.adaln_source == "hidden_mean":
            return x.mean(dim=1)
        return None

    # -- attach / detach -----------------------------------------------

    def attach(self, dit):
        if self._patches:
            return
        is_kv = getattr(self.ip, "kind", None) == "kv"

        for bidx, block in enumerate(dit.blocks):
            ca = block.cross_attn
            orig = ca.forward

            def make_forward(of, i, ca_mod):
                # comfy.ldm.cosmos.predict2.Attention.forward is
                #   (x, context=None, rope_emb=None, transformer_options={})
                # and the block calls it with transformer_options as a kwarg.
                # Take *args/**kwargs so signature drift can never break this.
                def new_forward(*args, **kwargs):
                    out = of(*args, **kwargs)
                    if not self._active():
                        return out
                    if is_kv and i >= self.ip.num_blocks:
                        return out

                    x = args[0] if args else kwargs["x"]
                    if is_kv:
                        bucket = [] if self.capture else None
                        ip_out = self.ip.forward_block(
                            i, x, ca_mod, self.image_tokens, self._cond_vec(x),
                            self.inject_point, self.reuse_k_norm, bucket,
                        )
                        if bucket:
                            prev = getattr(self.ip, "_attn_map", None)
                            n = getattr(self.ip, "_attn_n", 0)
                            cur = bucket[0].cpu()
                            self.ip._attn_map = cur if prev is None else prev + cur
                            self.ip._attn_n = n + 1
                    else:
                        ip_out = self.ip.forward_block(
                            i, x, self.image_tokens, scale_override=None
                        )
                    return out + self.weight * ip_out.to(out.dtype)

                return new_forward

            ca.forward = make_forward(orig, bidx, ca)
            self._patches.append((ca, orig))

    def detach(self):
        for ca, orig in self._patches:
            ca.forward = orig
        self._patches.clear()


# ─── ComfyUI Nodes ────────────────────────────────────────────────────

class AnimaIPAdapterLoader:
    """Load an Anima IP-Adapter from a safetensors checkpoint."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ipadapter": (folder_paths.get_filename_list("ipadapter"),),
            },
            "optional": {
                "lora_alpha": ("FLOAT", {
                    "default": 32.0, "min": 1.0, "max": 256.0, "step": 1.0,
                    "tooltip": "PEFT alpha for the co-trained LoRA. Not stored in "
                               "the safetensors; assume alpha == rank (32) unless "
                               "the training repo's adapter_config.json says otherwise.",
                }),
            },
        }

    RETURN_TYPES = ("ANIMA_IPADAPTER",)
    FUNCTION = "load"
    CATEGORY = "Farrenzo's Garbage/Anima/IP Adapter"

    def load(self, ipadapter, lora_alpha=32.0):
        from safetensors.torch import load_file

        path = folder_paths.get_full_path_or_raise("ipadapter", ipadapter)
        state = load_file(path)

        has_kv = any(k.startswith("blocks.") and ".ip_k_proj." in k for k in state)

        if "resampler.time_proj.weight" in state:
            # ---- v3: TimeResampler (InstantCharacter-style) ----
            num_blocks = max(
                int(k.split(".")[1]) for k in state if k.startswith("ip_cross_attns.")
            ) + 1
            num_queries = state["resampler.latents"].shape[1]
            resampler_dim = state["resampler.latents"].shape[2]
            input_dim = state["resampler.proj_in.weight"].shape[1]
            output_dim = state["resampler.proj_out.weight"].shape[0]
            module = IPAdapterSigLIP(
                input_dim=input_dim, dit_dim=output_dim, num_blocks=num_blocks,
                resampler_depth=len([
                    k for k in state
                    if k.startswith("resampler.layers.") and "adaLN" in k
                ]),
                num_queries=num_queries, resampler_dim=resampler_dim,
            )
            ip_state = state
            lora_patches = {}
            kind = "v3"
            n_blocks = num_blocks

        elif has_kv:
            # ---- KV / LuciferTC: decoupled K/V + co-trained LoRA ----
            n_blocks = max(
                int(k.split(".")[1]) for k in state
                if k.startswith("blocks.") and ".ip_k_proj." in k
            ) + 1
            dit_dim, img_dim = state["blocks.0.ip_k_proj.weight"].shape

            adaln_key = "blocks.0.adaln_ip.1.weight"
            has_adaln = adaln_key in state
            adaln_in = state[adaln_key].shape[1] if has_adaln else dit_dim

            module = AnimaIPAdapterKV(
                num_blocks=n_blocks, dit_dim=dit_dim, img_dim=img_dim,
                has_adaln=has_adaln, adaln_in=adaln_in,
            )
            ip_state = {k: v for k, v in state.items() if k.startswith("blocks.")}
            lora_patches = _build_lora_patches(state, alpha=float(lora_alpha))
            kind = "kv"

            print(
                f"[AnimaIPAdapter] KV checkpoint: {n_blocks} blocks, "
                f"dit_dim={dit_dim}, img_dim={img_dim}, "
                f"adaln={'yes' if has_adaln else 'no'} (in={adaln_in}), "
                f"{len(lora_patches)} LoRA targets"
            )
            if not lora_patches:
                print(
                    "[AnimaIPAdapter] WARNING: no co-trained LoRA found in this "
                    "checkpoint. If the adapter was trained with one, the IP "
                    "branch will underperform badly."
                )

        elif "resampler.latents" in state:
            raise RuntimeError(
                "v1/v2 checkpoint not compatible with the v3 architecture. "
                "Please retrain with the latest code."
            )
        else:
            raise KeyError(
                f"Unknown checkpoint format. Keys: {list(state.keys())[:10]}"
            )

        missing, unexpected = module.load_state_dict(ip_state, strict=False)
        if missing or unexpected:
            print(
                f"[AnimaIPAdapter] load_state_dict: {len(missing)} missing, "
                f"{len(unexpected)} unexpected"
            )
            for k in list(missing)[:8]:
                print(f"  missing:    {k}")
            for k in list(unexpected)[:8]:
                print(f"  unexpected: {k}")
        if missing:
            raise RuntimeError(
                f"IP-Adapter is missing {len(missing)} tensors after load — the "
                "reconstructed module does not match this checkpoint. Refusing to "
                "run a half-initialised adapter (see console for names)."
            )

        module.eval().requires_grad_(False)

        return ({
            "kind": kind,
            "module": module,
            "lora_patches": lora_patches,
            "ip_weights": ip_state,
            "num_blocks": n_blocks,
            "name": ipadapter,
        },)


class AnimaIPAdapterApply:
    """Apply an Anima IP-Adapter (and its co-trained LoRA) to a model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("ANIMA_IPADAPTER",),
                "siglip_features": ("SIGLIP_FEATURES", {
                    "tooltip": "SigLIP2 patch features [1, N, 768]"}),
                "weight": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "IP-Adapter strength"}),
                "lora_strength": ("FLOAT", {
                    "default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Strength of the co-trained LoRA. The IP branch was "
                               "trained with this active — 0.0 does not give you "
                               "'IP-Adapter only', it gives you a broken model."}),
                "start_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Step percent at which the IP branch switches on"}),
                "end_percent": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Step percent at which the IP branch switches off"}),
            },
            "optional": {
                "inject_point": (["direct", "shared_output_proj"], {
                    "default": "direct",
                    "tooltip": "UNKNOWN from weights. 'direct' adds the IP result "
                               "after the block's output_proj; 'shared_output_proj' "
                               "routes it through output_proj first (equivalent to "
                               "summing before the projection). Sweep both."}),
                "adaln_source": (["off", "hidden_mean"], {
                    "default": "off",
                    "tooltip": "UNKNOWN from weights. What drives adaln_ip. 'off' "
                               "skips the gate entirely. Sweep both."}),
                "capture_attention": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Record real cross-attention mass per image patch for "
                               "FG Anima IP Attention Map. Costs an extra attention "
                               "pass per block per step — diagnostic only."}),
                "reuse_k_norm": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run the IP keys through the block's own k_norm, as "
                               "the text branch does. Off feeds raw ip_k_proj output "
                               "against an RMSNorm'd query, which usually blows up "
                               "the attention scale. Leave on unless testing."}),
            },
        }

    RETURN_TYPES = ("MODEL", "ANIMA_IPADAPTER")
    RETURN_NAMES = ("model", "ipadapter")
    FUNCTION = "apply"
    CATEGORY = "Farrenzo's Garbage/Anima/IP Adapter"

    def apply(self, model, ipadapter, siglip_features, weight, lora_strength,
              start_percent, end_percent, inject_point="direct", adaln_source="off",
              reuse_k_norm=True, capture_attention=False):
        m = model.clone()

        module = ipadapter["module"]
        lora_patches = ipadapter.get("lora_patches") or {}

        if lora_patches and lora_strength != 0.0:
            loaded = m.add_patches(lora_patches, float(lora_strength))
            print(
                f"[AnimaIPAdapter] LoRA: {len(loaded)}/{len(lora_patches)} keys "
                f"matched at strength {lora_strength}"
            )
            if len(loaded) < len(lora_patches):
                unmatched = set(lora_patches) - set(loaded)
                for k in list(unmatched)[:8]:
                    print(f"  unmatched: {k}")
                print(
                    "[AnimaIPAdapter] WARNING: some LoRA targets did not match the "
                    "loaded DiT. Wrong Anima variant, or a module-name mismatch."
                )

        model_sampling = m.get_model_object("model_sampling")
        sigma_start = float(model_sampling.percent_to_sigma(start_percent))
        sigma_end = float(model_sampling.percent_to_sigma(end_percent))

        hook = IPAdapterHook(
            module, float(weight), sigma_start, sigma_end,
            inject_point=inject_point, adaln_source=adaln_source,
            reuse_k_norm=reuse_k_norm, capture=capture_attention,
        )
        if capture_attention:
            module._attn_map = None
            module._attn_n = 0
        hook.siglip_features = siglip_features

        def ip_wrapper(executor, *args, **kwargs):
            x = args[0]
            device, dtype = x.device, x.dtype

            module.to(device=device, dtype=dtype)

            feats = hook.siglip_features.to(device=device, dtype=dtype)
            if getattr(module, "kind", None) == "kv":
                # No resampler: raw SigLIP patch tokens go straight in.
                hook.image_tokens = feats
            else:
                hook.image_tokens = None  # set_timestep re-encodes per step

            timestep = args[1] if len(args) > 1 else kwargs.get("timesteps")
            hook.set_timestep(timestep)

            hook.attach(m.model.diffusion_model)
            try:
                return executor(*args, **kwargs)
            finally:
                hook.detach()

        m.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, IP_WRAPPER_KEY
        )
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            IP_WRAPPER_KEY,
            ip_wrapper,
        )
        return (m, ipadapter)


class AnimaSiglipeEncodeImage:
    """Extract SigLIP2 patch features from a reference image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("SIGLIP_FEATURES",)
    FUNCTION = "encode"
    CATEGORY = "Farrenzo's Garbage/Anima/IP Adapter"

    def encode(self, image):
        siglip, proc = _load_siglip2()
        device = mm.text_encoder_device()
        dtype = next(siglip.parameters()).dtype

        try:
            mm.free_memory(2 * 1024 ** 3, device)
        except Exception:
            pass
        siglip.to(device)

        batch = image if image.ndim == 4 else image.unsqueeze(0)

        padded = []
        for i in range(batch.shape[0]):
            img_np = (batch[i].cpu().numpy() * 255).clip(0, 255).astype("uint8")
            padded.append(prepare_siglip_image(PILImage.fromarray(img_np)))

        inputs = proc(images=padded, return_tensors="pt", do_resize=False)
        inputs = {k: v.to(device=device, dtype=dtype) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                features = siglip(**inputs).last_hidden_state  # [B, N, 768]
        finally:
            siglip.to(mm.text_encoder_offload_device())
            mm.soft_empty_cache()

        return (features.float(),)



class AnimaIPAdapterVisualize:
    """Visualize where the IP-Adapter is 'looking' in the reference image."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ipadapter": ("ANIMA_IPADAPTER",),
                "ref_image": ("IMAGE",),
                "mode": (["attention", "key_norm", "token_norm", "key_unique", "combined"],
                         {"default": "attention",
                          "tooltip": "'attention' shows real recorded cross-attention "
                                     "and needs a sample run first with "
                                     "capture_attention on. The others are static key "
                                     "statistics computed from the weights alone — "
                                     "they measure signal strength and distinctiveness, "
                                     "not where the model looked."}),
                "opacity": ("FLOAT", {"default": 0.6, "min": 0.1,
                                      "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "after": (ANY, {
                    "tooltip": "Ordering pin only; the value is ignored. Wire the "
                               "KSampler's LATENT (or the decoded IMAGE) here so this "
                               "node runs AFTER sampling. Without it ComfyUI may run "
                               "this branch first and there will be no attention to "
                               "show."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Attention is recorded as a side effect of sampling, so the inputs can be
        # identical while the data behind them is new. Never serve a cached result.
        return float("nan")

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("heatmap",)
    FUNCTION = "visualize"
    CATEGORY = "Farrenzo's Garbage/Anima/IP Adapter"

    def visualize(self, ipadapter, ref_image, mode, opacity, after=None):
        ip_weights = ipadapter["ip_weights"]
        num_blocks = ipadapter.get("num_blocks") or 0
        if not num_blocks:
            num_blocks = max(
                (int(k.split(".")[1]) for k in ip_weights if k.startswith("blocks.")),
                default=-1,
            ) + 1

        siglip, proc = _load_siglip2()
        device = mm.text_encoder_device()
        dtype = next(siglip.parameters()).dtype

        arr = (ref_image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = PILImage.fromarray(arr, mode="RGB")
        w0, h0 = pil_img.size
        square = prepare_siglip_image(pil_img)   # identical to the encoder's path

        if w0 != h0:
            waste = 1.0 - (min(w0, h0) / max(w0, h0))
            print(f"[AnimaIPAdapter] reference is {w0}x{h0}; letterboxed to 512x512, "
                  f"so ~{waste * 100:.0f}% of the 1024 patches are blank padding. "
                  f"Crop to square for full token budget.")

        siglip.to(device)
        try:
            inputs = proc(images=[square], return_tensors="pt", do_resize=False)
            inputs = {k: v.to(device=device, dtype=dtype) for k, v in inputs.items()}
            with torch.no_grad():
                ip_tokens = siglip(**inputs).last_hidden_state
        finally:
            siglip.to(mm.text_encoder_offload_device())
            mm.soft_empty_cache()

        recorded = getattr(ipadapter["module"], "_attn_map", None)
        rec_n = getattr(ipadapter["module"], "_attn_n", 0)
        if mode == "attention" and recorded is None:
            raise RuntimeError(
                "No attention recorded. All of these must hold:\n"
                "  1. capture_attention is ON in AnimaIPAdapterApply\n"
                "  2. weight > 0 — at weight 0.00 the IP branch is skipped entirely\n"
                "  3. start_percent/end_percent leave a non-empty window; the "
                "defaults are 0.0 and 1.0, and start=end=1.0 gates it to nothing\n"
                "  4. a KSampler using that MODEL has actually run this queue\n"
                "  5. this node's 'after' pin is wired downstream of that KSampler, "
                "or it may execute before sampling"
            )

        tokens = ip_tokens[0].float()
        n_tok = tokens.shape[0]
        side = int(round(n_tok ** 0.5))
        if side * side != n_tok:
            raise ValueError(
                f"SigLIP returned {n_tok} tokens, which is not a square grid; "
                "cannot build a spatial heatmap."
            )

        token_norm_map = tokens.norm(dim=1)
        key_norm_map = torch.zeros(n_tok, device=tokens.device)
        key_unique_map = torch.zeros(n_tok, device=tokens.device)
        used = 0

        for i in range(num_blocks):
            w_key = f"blocks.{i}.ip_k_proj.weight"
            if w_key not in ip_weights:
                continue
            wmat = ip_weights[w_key].float().to(tokens.device)
            keys = tokens @ wmat.T
            key_norm_map += keys.norm(dim=1)

            k_norm = F.normalize(keys, dim=1)
            sim = k_norm @ k_norm.T
            avg_cos = (sim.sum(dim=1) - 1.0) / max(1, n_tok - 1)
            key_unique_map += (1.0 - avg_cos)
            used += 1

        if used > 0:
            key_norm_map /= used
            key_unique_map /= used

        def minmax(t):
            return (t - t.min()) / (t.max() - t.min() + 1e-8)

        if mode == "attention":
            score_map = (recorded.to(tokens.device) / max(1, rec_n))[:n_tok]
        elif mode == "token_norm":
            score_map = token_norm_map
        elif mode == "key_norm":
            score_map = key_norm_map
        elif mode == "key_unique":
            score_map = key_unique_map
        else:
            score_map = (minmax(token_norm_map) + minmax(key_norm_map)
                         + minmax(key_unique_map)) / 3.0

        score_2d = score_map.detach().cpu().reshape(1, 1, side, side)
        score_up = F.interpolate(score_2d, size=(512, 512), mode="bilinear",
                                 align_corners=False).squeeze().numpy()

        smin, smax = float(score_up.min()), float(score_up.max())
        if smax - smin > 1e-8:
            score_up = (score_up - smin) / (smax - smin)

        heatmap_rgb = np.zeros((512, 512, 3), dtype=np.float32)
        for c in range(3):
            lo, mid, hi = [0.0, 0.0, 0.8][c], [0.0, 1.0, 0.0][c], [1.0, 0.0, 0.0][c]
            heatmap_rgb[:, :, c] = np.where(
                score_up < 0.5,
                lo + (mid - lo) * (score_up / 0.5),
                mid + (hi - mid) * ((score_up - 0.5) / 0.5),
            )

        ref_rgb = np.asarray(square, dtype=np.float32) / 255.0
        overlay = np.clip(heatmap_rgb * opacity + ref_rgb * (1.0 - opacity), 0.0, 1.0)
        result = torch.from_numpy(overlay).float().unsqueeze(0)

        if mode == "attention":
            top = score_map.topk(min(8, n_tok)).values
            print(f"[AnimaIPAdapter] attention over {rec_n} block-steps; "
                  f"uniform would be {1.0 / n_tok:.5f}, top-8 mean {top.mean():.5f} "
                  f"(ratio {top.mean() * n_tok:.1f}x). Ratio near 1.0 means the IP "
                  f"branch is attending to nothing in particular.")

        print(f"[AnimaIPAdapter] Heatmap: mode={mode}, blocks_used={used}, "
              f"token_norm={token_norm_map.mean().item():.4f}, "
              f"key_norm={key_norm_map.mean().item():.4f}, "
              f"key_unique={key_unique_map.mean().item():.4f}")

        return (result,)
