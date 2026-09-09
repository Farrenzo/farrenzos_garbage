"""
FG_ConditioningCache
---------------------
Save/Load CONDITIONING + LATENT to/from disk together, so a heavy text
encoder (e.g. Qwen3-VL-32B for MiniMax H3) can be fully evicted from VRAM
after encoding, and a later run can skip the encoder entirely by loading
the cached pair straight into the sampler.

Conditioning and latent are saved as one file so they can never get
mismatched (e.g. loading conditioning from one prompt against a latent
from another run).

Shares fg_helperfunctions.log() and .clear_memory() / get_output_path()
so the "empty the card" logic lives in one place other nodes can reuse.
"""

import os
import torch
import folder_paths
from ._fg_helperfunctions import log, clear_memory, get_output_path


category = "Farrenzo's Garbage/Conditioning"
save_description = "Embedd conditioning and save it in a pickle file."
load_description = "Load a previously saved pickle file with conditioning embedded within it."


def _cache_root():
    return os.path.join(folder_paths.get_output_directory(), "fg_conditioning_cache")


def _tensor_dict_to_cpu(d):
    """Clone every tensor value in a dict to CPU; pass non-tensor values through."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().to("cpu").clone()
        else:
            out[k] = v
    return out


def _conditioning_to_cpu(conditioning):
    out = []
    for emb, meta in conditioning:
        emb_cpu = emb.detach().to("cpu").clone()
        out.append([emb_cpu, _tensor_dict_to_cpu(meta)])
    return out


def _latent_to_cpu(latent):
    # LATENT is a dict, typically {"samples": tensor, ...possible extra keys
    # like noise_mask / batch_index}. Handled the same generic way, so an
    # empty placeholder latent (all-zero samples tensor) round-trips fine.
    return _tensor_dict_to_cpu(latent)


class FG_SaveConditioningLatent:
    """
    Saves a CONDITIONING + LATENT pair to disk (one file, so they can't get
    mismatched later) and passes both through unchanged. Optionally
    force-unloads all currently loaded models + does a nuclear cache purge
    right after saving, so the text encoder doesn't linger in VRAM for the
    rest of the workflow.
    """
    def __init__(self):
        self.NODE_NAME = "Save Conditioning"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_conditioning": ("CONDITIONING",),
                "filename_prefix": ("STRING", {"default": "cond_cache_%HMSf%"}),
                "force_unload_after_save": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "negative_conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("+ conditioning", "- conditioning", "latent")
    FUNCTION = "save"
    CATEGORY = category
    DESCRIPTION = save_description
    OUTPUT_NODE = True

    def save(self, positive_conditioning, filename_prefix, force_unload_after_save, negative_conditioning=None, latent=None):
        full_folder, filename_base, _ = get_output_path(
            node_name="FG_SaveConditioningLatent",
            filename_prefix=filename_prefix,
            output_path=_cache_root(),
        )
        path = os.path.join(full_folder, f"{filename_base}.pt")

        payload = {
            "positive_conditioning": _conditioning_to_cpu(positive_conditioning),
            "negative_conditioning": _conditioning_to_cpu(negative_conditioning) if negative_conditioning else [],
            "latent": _latent_to_cpu(latent) if latent else {"samples": None},
        }
        print(type(payload["positive_conditioning"]))

        torch.save(payload, path)
        log(f"💾 {self.NODE_NAME} and/or Latent To Disk -> {path}", message_type="finish")

        if force_unload_after_save:
            # Nuclear: models AND their VRAM footprint, not just a "soft"
            # unload that leaves weights resident. See fg_helperfunctions
            # .clear_memory() for why the plain unload_all_models() call
            # alone wasn't enough on this rig.
            clear_memory(purge_cache=True, purge_models=True, nuclear=True)

        return (positive_conditioning, negative_conditioning, latent)


class FG_LoadConditioningLatent:
    """
    Loads a previously saved CONDITIONING + LATENT pair from disk. Wire this
    in place of the live CLIP/text-encoder branch (and its empty-latent
    source) when you want to resume or reuse a prompt without paying to
    reload the encoder.
    """
    def __init__(self):
        self.NODE_NAME = "Load Conditioning"

    @classmethod
    def INPUT_TYPES(cls):
        d = _cache_root()
        os.makedirs(d, exist_ok=True)
        files = sorted(
            (f[:-3] for f in os.listdir(d) if f.endswith(".pt")),
            reverse=True,
        )
        return {
            "required": {
                "filename": (files if files else ["<no cached files found>"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("+ Conditioning", "- Conditioning", "Latent")
    FUNCTION = "load"
    DESCRIPTION = load_description
    CATEGORY = category

    def load(self, filename):
        path = os.path.join(_cache_root(), f"{filename}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[FG_LoadConditioningLatent] No cached file at {path}. "
                f"Run FG_SaveConditioningLatent first."
            )

        # weights_only=False is required: conditioning metadata dicts can
        # carry non-tensor entries. This is your own locally generated
        # file, not an untrusted download, so this is safe.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        log(f"📂 {self.NODE_NAME}: Retrieved conditioning data from pickle <- {path}", message_type="finish")

        return (payload["positive_conditioning"], payload["negative_conditioning"], payload["latent"])


