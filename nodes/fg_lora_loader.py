"""
┌─────────────────────────────────────┐
│          Multi LoRA Loader          │
├─────────────────────────────────────┤
│ ○ Model                    Model  ○ │
│ ○ Clip                      Clip  ○ │
│                                     │
│          + Add LoRA Button          │
│                                     │
└─────────────────────────────────────┘

The per-LoRA rows are drawn by web/js/fg_dynamic_lora_loader.js, which
writes them into the hidden lora_stack widget as a JSON array.
"""
import json
import os
import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import io
from ._fg_helperfunctions import log

NODE_NAME = "FG_DynamicLoraLoader"

global_description = """
Dynamic LoRA stacker with optional CLIP entry.
 - Can be used with SDXL models. (CLIP required)
 - FLUX models. (CLIP optional)
"""


class FG_LoraLoader(io.ComfyNode):
    """LoRA loader with dynamic add/remove buttons. CLIP input is optional."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=NODE_NAME,
            display_name="🗑️ Multi-LoRA Loader",
            category="Farrenzo's Garbage/Model Loaders",
            description=global_description,
            inputs=[
                io.Model.Input("model", tooltip="🤖 Model: Diffusion Model input used for generation"),
                io.Clip.Input("clip", optional=True, tooltip="📒 Clip: used for text encoding and conditional generation. This is optional."),
                io.String.Input("lora_stack", default="[]", optional=True, tooltip="JSON array of LoRA configs, managed by the node UI."),
            ],
            outputs=[
                io.Model.Output(display_name="Model"),
                io.Clip.Output(display_name="Clip"),
            ],
        )

    @classmethod
    def execute(cls, model, clip=None, lora_stack="[]") -> io.NodeOutput:
        loaded = []
        for lora_config in json.loads(lora_stack):
            name = lora_config.get("name")
            if not name:
                continue

            strength_model = float(lora_config.get("strength_model", 1.0))
            strength_clip = float(lora_config.get("strength_clip", 1.0)) if clip is not None else 0
            if strength_model == 0 and strength_clip == 0:
                continue

            lora_path = folder_paths.get_full_path_or_raise("loras", name)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model, clip = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
            loaded.append(os.path.basename(lora_path))

        if loaded:
            log(f"{NODE_NAME}: Total LoRA's: {len(loaded)}")
            for lora_name in loaded:
                log(f"{NODE_NAME}: Loaded: {lora_name}")
        else:
            log(f"{NODE_NAME}: No LoRA's loaded.")

        return io.NodeOutput(model, clip)
