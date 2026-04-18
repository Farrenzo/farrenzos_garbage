"""
Batch Conditioning Node
┌──────────────────────────────────────────┐
│         Batch Conditioning               │
├──────────────────────────────────────────┤
│ ○ CLIP (optional)      + Positive    ○   │
│ ○ VAE  (optional)      - Negative    ○   │
│                        Image Path    ○   │
│ <→ INPUT> JSON Path    Total Count   ○   │
│ <↔ SWITCH> Create/Load                   │
│ <↔ BOOL> Zero Out Negative               │
│ <→ INPUT> Index                          │
│                                          │
└──────────────────────────────────────────┘

Create Mode:
  - Requires CLIP (and optionally VAE)
  - Parses JSON for image paths and captions
  - Encodes ALL entries and saves conditioning to safetensors
  - Updates JSON with conditioning file paths
  - Outputs the first entry's conditioning

Load Mode:
  - No CLIP/VAE needed
  - Loads conditioning at the given index
  - Outputs conditioning + image path for downstream nodes
"""

import os
import json
import math
import torch
import numpy as np
import comfy.utils
import node_helpers
from PIL import Image
from safetensors.torch import save_file
from safetensors import safe_open
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from ._fg_helperfunctions import log

global_instruction_content = (
    "Describe the key features of the input image (color, shape, size, "
    "texture, objects, background), then explain how the user's text "
    "instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency "
    "with the original input where appropriate."
)


class FG_BatchConditioning(ComfyNodeABC):

    def __init__(self):
        self.NODE_NAME = "Batch Conditioning"

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "json_filepath": (IO.STRING, {
                    "default": "",
                    "tooltip": "Path to the JSON batch file."
                }),
                "mode": (["create", "load"], {
                    "default": "create",
                    "tooltip": "Create: encode and save all conditionings. Load: load one at a time by index."
                }),
                "index": (IO.INT, {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Which entry to output (load mode). Ignored in create mode."
                }),
                "zero_out_negative": (IO.BOOLEAN, {
                    "default": True,
                    "label_on": "Zero out negative",
                    "label_off": "Encode empty negative"
                }),
            },
            "optional": {
                "clip": (IO.CLIP, {
                    "tooltip": "🟡 Required for create mode. The CLIP/VL model for encoding."
                }),
                "vae": (IO.VAE, {
                    "tooltip": "🎨 Optional. Encodes reference latents for image-guided generation."
                }),
                "vl_instruction": ("STRING", {
                    "multiline": True,
                    "default": global_instruction_content,
                    "placeholder": global_instruction_content,
                    "tooltip": "📝 System instructions for VL-based CLIP models."
                }),
            }
        }

    RETURN_TYPES  = ("CONDITIONING", "CONDITIONING", "STRING", "INT")
    RETURN_NAMES  = ("positive", "negative", "image_path", "total")
    OUTPUT_TOOLTIPS = (
        "➕ Positive conditioning",
        "➖ Negative conditioning",
        "Path to the current image (for loading into VAE/scaler)",
        "Total number of entries in the JSON",
    )
    FUNCTION = "process"
    CATEGORY = "Farrenzo's Garbage/Conditioning"
    DESCRIPTION = "Batch-encode or batch-load conditionings from a JSON manifest. Create mode encodes all entries and saves to safetensors. Load mode retrieves one entry at a time by index."

    # ── Conditioning ↔ Safetensors ────────────────────────────────

    def _save_conditioning(self, conditioning, filepath):
        """Flatten a ComfyUI conditioning list into safetensors."""
        tensors  = {}
        metadata = {}

        for idx, (embed, attrs) in enumerate(conditioning):
            tensors[f"embed_{idx}"] = embed.cpu().contiguous()
            attr_keys = []

            for key, val in attrs.items():
                if isinstance(val, torch.Tensor):
                    tensors[f"attr_{idx}_{key}"] = val.cpu().contiguous()
                    attr_keys.append(key)
                elif isinstance(val, list):
                    count = 0
                    for j, item in enumerate(val):
                        if isinstance(item, torch.Tensor):
                            tensors[f"attr_{idx}_{key}_{j}"] = item.cpu().contiguous()
                            count += 1
                    metadata[f"attr_{idx}_{key}_count"] = str(count)
                    metadata[f"attr_{idx}_{key}_type"]  = "list"
                    attr_keys.append(key)

            metadata[f"embed_{idx}_keys"] = json.dumps(attr_keys)

        metadata["count"] = str(len(conditioning))
        save_file(tensors, filepath, metadata=metadata)

    def _load_conditioning(self, filepath, device="cpu"):
        """Reconstruct a ComfyUI conditioning list from safetensors."""
        tensors = {}
        with safe_open(filepath, framework="pt", device=device) as f:
            metadata = f.metadata()
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        count = int(metadata["count"])
        conditioning = []

        for idx in range(count):
            embed = tensors[f"embed_{idx}"]
            keys  = json.loads(metadata[f"embed_{idx}_keys"])
            attrs = {}

            for key in keys:
                list_count_key = f"attr_{idx}_{key}_count"
                if list_count_key in metadata:
                    lat_count = int(metadata[list_count_key])
                    attrs[key] = [
                        tensors[f"attr_{idx}_{key}_{j}"]
                        for j in range(lat_count)
                    ]
                elif f"attr_{idx}_{key}" in tensors:
                    attrs[key] = tensors[f"attr_{idx}_{key}"]

            conditioning.append([embed, attrs])

        return conditioning

    # ── CLIP / VL encoding ────────────────────────────────────────

    def _load_image_from_disk(self, image_path):
        """Load an image file and return a ComfyUI image tensor (1, H, W, 3)."""
        img = Image.open(image_path).convert("RGB")
        return torch.from_numpy(
            np.array(img).astype(np.float32) / 255.0
        ).unsqueeze(0)

    def _encode_prompt(self, clip, vae, prompt, image_tensor, vl_instruction):
        """Encode a prompt + optional image into conditioning (mirrors FG_CLIPTextEncode logic)."""
        images_vl   = []
        ref_latents = []
        llama_template = (
            f"<|im_start|>system\n{vl_instruction}<|im_end|>\n"
            f"<|im_start|>user\n{{}} <|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        image_prompt = ""

        if image_tensor is not None:
            image_prompt = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>\n"
            samples = image_tensor.movedim(-1, 1)

            # VL path — scale to ~384×384 pixel area
            total_px = int(384 * 384)
            scale_by = math.sqrt(total_px / (samples.shape[3] * samples.shape[2]))
            width  = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)
            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            images_vl.append(s.movedim(1, -1))

            # VAE ref_latent path — keep original resolution, round to mult of 8
            if vae is not None:
                width  = (samples.shape[3] + 7) // 8 * 8
                height = (samples.shape[2] + 7) // 8 * 8
                s = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
                ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

        if image_prompt == "":
            tokens = clip.tokenize(prompt)
        else:
            tokens = clip.tokenize(
                image_prompt + prompt,
                images=images_vl,
                llama_template=llama_template
            )

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"reference_latents": ref_latents},
                append=True
            )
        return conditioning

    def _zero_out(self, conditioning):
        """Zero out all tensors in a conditioning (for models that don't use negative prompts)."""
        c = []
        for t in conditioning:
            d = t[1].copy()
            pooled = d.get("pooled_output", None)
            if pooled is not None:
                d["pooled_output"] = torch.zeros_like(pooled)
            lyrics = d.get("conditioning_lyrics", None)
            if lyrics is not None:
                d["conditioning_lyrics"] = torch.zeros_like(lyrics)
            c.append([torch.zeros_like(t[0]), d])
        return c

    # ── Main entry point ──────────────────────────────────────────

    def process(
        self,
        json_filepath,
        mode,
        index,
        zero_out_negative,
        clip=None,
        vae=None,
        vl_instruction=global_instruction_content,
    ):
        if not os.path.isfile(json_filepath):
            raise RuntimeError(f"{self.NODE_NAME}: JSON file not found: {json_filepath}")

        with open(json_filepath, "r", encoding="utf-8") as f:
            batch_data = json.load(f)

        entries = list(batch_data.items())
        total   = len(entries)

        if total == 0:
            raise RuntimeError(f"{self.NODE_NAME}: JSON file has no entries.")

        # ── CREATE MODE ───────────────────────────────────────────

        if mode == "create":
            if clip is None:
                raise RuntimeError(f"{self.NODE_NAME}: CLIP is required for create mode.")

            cond_dir = os.path.join(os.path.dirname(json_filepath), "conditioning")
            os.makedirs(cond_dir, exist_ok=True)

            for i, (img_path, entry_data) in enumerate(entries):
                caption      = entry_data.get("caption", "")
                img_basename = os.path.splitext(os.path.basename(img_path))[0]

                # Load image from disk
                if os.path.isfile(img_path):
                    image_tensor = self._load_image_from_disk(img_path)
                else:
                    log(f"{self.NODE_NAME}: Image not found: {img_path}, encoding text-only.", "warning")
                    image_tensor = None

                # Encode positive
                positive = self._encode_prompt(clip, vae, caption, image_tensor, vl_instruction)

                # Encode negative
                if zero_out_negative:
                    negative = self._zero_out(positive)
                else:
                    negative = self._encode_prompt(clip, vae, "", image_tensor, vl_instruction)

                # Save to safetensors
                pos_path = os.path.join(cond_dir, f"{img_basename}_positive.safetensors")
                neg_path = os.path.join(cond_dir, f"{img_basename}_negative.safetensors")
                self._save_conditioning(positive, pos_path)
                self._save_conditioning(negative, neg_path)

                # Update JSON entry
                entry_data["conditioning"] = {
                    "positive": pos_path,
                    "negative": neg_path,
                }

                log(f"{self.NODE_NAME}: [{i + 1}/{total}] Saved conditioning for {img_basename}")

            # Write updated JSON
            with open(json_filepath, "w", encoding="utf-8") as f:
                json.dump(batch_data, f, indent=4)

            log(f"{self.NODE_NAME}: Created {total} conditioning pairs.", "finish")

            # Output the first entry so the graph can continue
            first_path = entries[0][0]
            first_data = entries[0][1]["conditioning"]
            positive = self._load_conditioning(first_data["positive"])
            negative = self._load_conditioning(first_data["negative"])
            return (positive, negative, first_path, total)

        # ── LOAD MODE ─────────────────────────────────────────────

        else:
            if index >= total:
                raise RuntimeError(
                    f"{self.NODE_NAME}: Index {index} out of range. Total entries: {total}"
                )

            img_path, entry_data = entries[index]
            cond_paths = entry_data.get("conditioning", {})

            pos_path = cond_paths.get("positive", "")
            neg_path = cond_paths.get("negative", "")

            if not pos_path or not os.path.isfile(pos_path):
                raise RuntimeError(
                    f"{self.NODE_NAME}: Positive conditioning not found for index {index}. "
                    f"Run create mode first."
                )
            if not neg_path or not os.path.isfile(neg_path):
                raise RuntimeError(
                    f"{self.NODE_NAME}: Negative conditioning not found for index {index}. "
                    f"Run create mode first."
                )

            positive = self._load_conditioning(pos_path)
            negative = self._load_conditioning(neg_path)

            log(f"{self.NODE_NAME}: Loaded conditioning [{index + 1}/{total}] — {os.path.basename(img_path)}")
            return (positive, negative, img_path, total)
