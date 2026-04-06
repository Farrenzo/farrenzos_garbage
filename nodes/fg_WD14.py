# import warnings
# warnings.filterwarnings("ignore", message=".*GenerationMixin.*")
# warnings.filterwarnings("ignore", message=".*generation flags.*")

import os
import torch
import numpy as np
from PIL import Image
from ._fg_helperfunctions import log

from .. import WD_14_INFO
WD14_MODEL_PATH = os.path.join(os.path.dirname(__file__), f"..\\models\\{WD_14_INFO['directory']}")

# Lazy imports to avoid loading everything at startup
ort = None
pd = None

def get_onnx():
    global ort
    if ort is None:
        import onnxruntime as ort_module
        ort = ort_module
    return ort


def get_pandas():
    global pd
    if pd is None:
        import pandas as pd_module
        pd = pd_module
    return pd

set_up_info = """
Because this node pack strongly believes in you downloading things for yourself
you are going to have to download the WD_1.4 booru tagger model. Go to:
https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/tree/main
Download: model.onnx
SHA256: 9e768793060c7939b277ccb382783e8670e8a042d29d77aa736be0c8cc898bfc
Place it in: custom_nodes/farrenzos_garbage/models/wd14_v3/THE_MODEL_YOU_DOWNLOADED
Restart comfy & Voila.
"""
class WD14Tagger:
    """
    Booru-style image tagger using WD14 v3 models.
    Outputs comma-separated tags suitable for anime/illustration prompts.
    """

    TAGGERS = WD_14_INFO["tagging_models"]

    # Cache for loaded models and tags
    _sessions = {}
    _tags = {}

    def __init__(self):
        self.NODE_NAME = "WD14 Tagger (Booru Tags)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image you would like tagged."}),
                "model": (list(cls.TAGGERS.keys()), {"default": "eva02-large", "tooltip": "The model you would like to do the tagging."}),
                "threshold": ("FLOAT", {
                    "default": 0.35,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "replace_underscores": ("BOOLEAN", {"default": True, "tooltip": "True to remove underscores, best for humans."}),
                "exclude_rating_tags": ("BOOLEAN", {"default": True, "tooltip": "True to remove the rating of the tag, also best for humans."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    OUTPUT_TOOLTIPS = (
        "The tags in a single line, comma separated.",
        "A list of tags and their scores.",
    )
    RETURN_NAMES = ("Tags", "Scored Tags", "Organized Prompt", )
    FUNCTION = "tag_image"
    CATEGORY = "Farrenzo's Garbage/Image/Tag Analysis"
    DESCRIPTION = "Loads a tagging model. Typically WD14."

    def _load_model(self, model_key: str):
        """Load and cache ONNX model and tags CSV."""
        if model_key not in self._sessions:
            ort = get_onnx()
            pd = get_pandas()
            
            config = self.TAGGERS[model_key]
            model_path = os.path.join(WD14_MODEL_PATH, config["model"])
            csv_path = os.path.join(WD14_MODEL_PATH, config["csv"])
            
            if not os.path.exists(model_path):
                log(f"Model not found: {model_path}", "error")
                log(f"{set_up_info}", "warning")
                raise FileNotFoundError(f"Model not found: {model_path}")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Tags CSV not found: {csv_path}")
            
            session = ort.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self._sessions[model_key] = session
            
            tags_df = pd.read_csv(csv_path)
            self._tags[model_key] = {
                "names": tags_df["name"].tolist(),
                "categories": tags_df["human_category"].tolist() if "human_category" in tags_df.columns else None
            }
        
        return self._sessions[model_key], self._tags[model_key]

    def _preprocess_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Convert ComfyUI IMAGE tensor to WD14 input format."""
        # ComfyUI IMAGE is (B, H, W, C) with values 0-1
        # Take first image if batch
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        
        # Convert to PIL
        img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_np, mode="RGB")
        
        # Letterbox to 448x448 square (don't stretch)
        size = 448
        max_dim = max(img.size)
        scale = size / max_dim
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Pad with white background
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        paste_x = (size - new_w) // 2
        paste_y = (size - new_h) // 2
        canvas.paste(img, (paste_x, paste_y))
        
        # Convert to numpy, BGR order, float32
        img_array = np.asarray(canvas).astype(np.float32)
        img_array = img_array[:, :, ::-1]  # RGB -> BGR
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

    def _prompt_builder(self, prompt_tags: dict[str, tuple[str, float]]) -> str:
        prompt = ""
        tags = dict(sorted(prompt_tags.items()))
        for category, tag_info in tags.items():
            prompt += f"{category}: {', '.join(t[0] for t in tag_info)}\n"
        return prompt

    def tag_image(self, image, model, threshold, replace_underscores, exclude_rating_tags):
        session, tags_data = self._load_model(model)
        tag_names  :list[str] = tags_data["names"]
        categories :list[str] = tags_data["categories"]
        
        input_name = session.get_inputs()[0].name
        img_array = self._preprocess_image(image)
        
        outputs = session.run(None, {input_name: img_array})
        scores = outputs[0][0]
        
        prompt_builder_results = {}
        for cat in set(categories):
            if exclude_rating_tags:
                if cat == "0_general":
                    continue
            prompt_builder_results[cat] = [
                (
                    tag_names[i].replace("_", " ") if replace_underscores else tag_names[i],
                    score
                ) for i, score in enumerate(scores) if score >= threshold and categories[i] == cat
            ]
            if len(prompt_builder_results[cat]) == 0:
                prompt_builder_results.pop(cat)

        results = []
        results_with_scores = []
        for cat, tag_info in prompt_builder_results.items():
            for tag in tag_info:
                results += [tag[0]]
                results_with_scores += [f"{tag[1]:.2f}: {tag[0]}"]

        tags_str = ", ".join(results)
        tags_with_scores_str = "\n".join(sorted(results_with_scores))
        prompt_string = self._prompt_builder(prompt_builder_results)
        
        return (tags_str, tags_with_scores_str, prompt_string, )
