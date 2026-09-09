import warnings
warnings.filterwarnings("ignore", message=".*GenerationMixin.*")
warnings.filterwarnings("ignore", message=".*generation flags.*")

import os
import numpy as np
import torch
from PIL import Image

from ._fg_helperfunctions import log
from .. import WD_14_INFO

# Lazy imports to avoid loading everything at startup
pd = None
ort = None


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

# Preference order when device="gpu". DirectML first: it covers NVIDIA and Intel
# on Windows with one package, which matters on a mixed-GPU box.
_GPU_PROVIDERS = (
    "DmlExecutionProvider",        # onnxruntime-directml
    "CUDAExecutionProvider",       # onnxruntime-gpu
    "OpenVINOExecutionProvider",   # onnxruntime-openvino
    "ROCMExecutionProvider",       # onnxruntime-rocm
)

_INSTALL_HINT = """No GPU execution provider is available.

The plain `onnxruntime` package is CPU-only, and passing CUDAExecutionProvider to
it does not error -- ORT warns and silently falls back to CPU. That is almost
certainly what has been happening.

These all install into the same `onnxruntime` namespace and conflict, so
uninstall first:

    pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml onnxruntime-openvino

then pick ONE:

    pip install onnxruntime-directml    # NVIDIA + Intel Arc on Windows (recommended here)
    pip install onnxruntime-gpu         # NVIDIA only, needs CUDA + cuDNN
    pip install onnxruntime-openvino    # Intel Arc, best Arc throughput

Available right now: {available}"""


def _resolve_providers(device: str, device_id: int):
    """Return (provider_list, human_readable_name). Raises if gpu is unavailable."""
    o = get_onnx()
    available = o.get_available_providers()

    if device == "cpu":
        return ["CPUExecutionProvider"], "CPUExecutionProvider"

    gpu = next((p for p in _GPU_PROVIDERS if p in available), None)

    if gpu is None:
        if device == "gpu":
            raise RuntimeError(_INSTALL_HINT.format(available=available))
        return ["CPUExecutionProvider"], "CPUExecutionProvider (no GPU provider installed)"

    if gpu == "OpenVINOExecutionProvider":
        opts = {"device_type": "GPU"}
    else:
        opts = {"device_id": int(device_id)}
    return [(gpu, opts), "CPUExecutionProvider"], f"{gpu} (device_id={device_id})"


class FG_WD14Tagger:
    """
    Booru-style image tagger using WD14 v3 models.
    Outputs comma-separated tags suitable for anime/illustration prompts.
    """

    TAGGERS = WD_14_INFO["tagging_models"]

    # Cache keyed by (model, device, device_id) so switching device rebuilds.
    _sessions = {}
    _tags = {}

    def __init__(self):
        self.NODE_NAME = "WD14 Tagger (Booru Tags)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image you would like tagged."}),
                "model": (list(cls.TAGGERS.keys()), {
                    "default": "eva02-large",
                    "tooltip": "The model you would like to do the tagging."}),
                "device": (["auto", "gpu", "cpu"], {
                    "default": "auto",
                    "tooltip": "auto = GPU when a GPU provider is installed, else CPU "
                               "silently. gpu = raise a useful error instead of "
                               "falling back. On a ~1GB tagger CPU is often within a "
                               "few hundred ms of GPU, so measure before switching."}),
                "threshold": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                    "display": "slider",
                    "tooltip": "Confidence floor for general tags."}),
                "replace_underscores": ("BOOLEAN", {
                    "default": True, "tooltip": "True to remove underscores, best for humans."}),
                "exclude_rating_tags": ("BOOLEAN", {
                    "default": True, "tooltip": "True to remove the rating of the tag, also best for humans."}),
            },
            "optional": {
                "character_threshold": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
                    "display": "slider",
                    "tooltip": "Separate, higher floor for character tags. They are "
                               "confidently wrong far more often than general tags, "
                               "and one bad character name derails a whole prompt."}),
                "device_id": ("INT", {
                    "default": 0, "min": 0, "max": 8,
                    "tooltip": "Which GPU, when more than one is present. Ignored on CPU."}),
                "sort_by_score": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Order the Tags output most-confident first rather "
                               "than by CSV row order."}),
                "exclude_tags": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Comma-separated tags to drop, matched after "
                               "underscore replacement."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("Tags", "Scored Tags", "Organized Prompt", "Info")
    OUTPUT_TOOLTIPS = (
        "The tags in a single line, comma separated.",
        "A list of tags and their scores.",
        "Tags grouped by category, one line each.",
        "Which execution provider actually ran, and timings.",
    )
    FUNCTION = "tag_image"
    CATEGORY = "Farrenzo's Garbage/Image/Utils"
    DESCRIPTION = "Loads a tagging model. Typically WD14."

    # -- model ---------------------------------------------------------

    def _load_model(self, model_key: str, device: str, device_id: int):
        cache_key = (model_key, device, device_id)
        if cache_key not in self._sessions:
            o = get_onnx()
            pdm = get_pandas()

            config = self.TAGGERS[model_key]
            model_path = os.path.join(WD_14_INFO["model_path"], config["model"])
            csv_path = os.path.join(WD_14_INFO["model_path"], config["csv"])

            if not os.path.exists(model_path):
                log(f"Model not found: {model_path}", "error")
                log(f"{set_up_info}", "warning")
                raise FileNotFoundError(f"Model not found: {model_path}")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Tags CSV not found: {csv_path}")

            providers, label = _resolve_providers(device, device_id)

            # ORT's default CPU arena reserves a large block and never returns it,
            # which can push ComfyUI's memory manager into unloading models that
            # had nothing to do with this node.
            so = o.SessionOptions()
            so.enable_cpu_mem_arena = False
            so.enable_mem_pattern = False

            session = o.InferenceSession(model_path, sess_options=so, providers=providers)
            actual = session.get_providers()
            log(f"WD14 '{model_key}' requested {label}; running on {actual[0]}")
            if device == "gpu" and actual[0] == "CPUExecutionProvider":
                raise RuntimeError(
                    f"Requested GPU but ORT fell back to CPU. Available: "
                    f"{o.get_available_providers()}"
                )

            self._sessions[cache_key] = session

            tags_df = pdm.read_csv(csv_path)
            cats = (tags_df["human_category"].tolist()
                    if "human_category" in tags_df.columns else None)
            if cats is None and "category" in tags_df.columns:
                # Standard SmilingWolf CSV: 0 general, 4 character, 9 rating.
                lut = {0: "00_GENERAL", 4: "02_CHARACTER", 9: "09_RATING"}
                cats = [lut.get(int(c), f"{int(c):02d}_OTHER") for c in tags_df["category"]]
            if cats is None:
                cats = ["00_GENERAL"] * len(tags_df)

            self._tags[cache_key] = {"names": tags_df["name"].tolist(), "categories": cats}

        return self._sessions[cache_key], self._tags[cache_key]

    @classmethod
    def unload(cls):
        """Free every cached ORT session. Wire into FG_PurgeVRAM."""
        cls._sessions.clear()
        cls._tags.clear()

    # -- preprocessing --------------------------------------------------

    def _preprocess_image(self, image_tensor: torch.Tensor, size: int) -> np.ndarray:
        if image_tensor.ndim == 4:
            image_tensor = image_tensor[0]
        # Some nodes hand back RGBA; Image.fromarray(..., "RGB") would throw.
        image_tensor = image_tensor[..., :3]

        img_np = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_np, mode="RGB")

        # Letterbox to a square on white (don't stretch)
        scale = size / max(img.size)
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        canvas.paste(img, ((size - new_w) // 2, (size - new_h) // 2))

        arr = np.asarray(canvas).astype(np.float32)
        arr = arr[:, :, ::-1]  # RGB -> BGR, what SmilingWolf's models expect
        # ::-1 leaves a negative stride; ORT requires a contiguous buffer.
        return np.ascontiguousarray(np.expand_dims(arr, axis=0))

    def _prompt_builder(self, prompt_tags) -> str:
        prompt = ""
        for category, tag_info in sorted(prompt_tags.items()):
            prompt += f"{category}: {', '.join(t[0] for t in tag_info)}\n"
        return prompt

    # -- node -----------------------------------------------------------

    def tag_image(self, image, model, device, threshold, replace_underscores,
                  exclude_rating_tags, character_threshold=0.85, device_id=0,
                  sort_by_score=True, exclude_tags=""):
        import time

        t0 = time.perf_counter()
        session, tags_data = self._load_model(model, device, device_id)
        t_load = time.perf_counter() - t0

        tag_names = tags_data["names"]
        categories = tags_data["categories"]

        inp = session.get_inputs()[0]
        # Input is NHWC; read the side length from the graph instead of hardcoding
        # 448, since model versions differ.
        size = next((d for d in inp.shape[1:3] if isinstance(d, int)), 448)

        img_array = self._preprocess_image(image, size)

        t1 = time.perf_counter()
        outputs = session.run(None, {inp.name: img_array})
        t_infer = time.perf_counter() - t1
        scores = outputs[0][0]

        drop = {t.strip().lower() for t in exclude_tags.split(",") if t.strip()}

        grouped = {}
        for i, score in enumerate(scores):
            cat = categories[i]
            if exclude_rating_tags and ("RATING" in cat.upper() or cat == "01_META"):
                continue
            floor = character_threshold if "CHARACTER" in cat.upper() else threshold
            if score < floor:
                continue
            name = tag_names[i].replace("_", " ") if replace_underscores else tag_names[i]
            if name.lower() in drop:
                continue
            grouped.setdefault(cat, []).append((name, float(score)))

        for cat in grouped:
            grouped[cat].sort(key=lambda t: -t[1])

        flat = [t for cat in sorted(grouped) for t in grouped[cat]]
        if sort_by_score:
            flat = sorted(flat, key=lambda t: -t[1])

        tags_str = ", ".join(t[0] for t in flat)
        scored_str = "\n".join(f"{s:.2f}: {n}" for n, s in
                               sorted(flat, key=lambda t: -t[1]))
        prompt_string = self._prompt_builder(grouped)

        info = (f"provider: {session.get_providers()[0]}\n"
                f"input   : {size}x{size} NHWC BGR\n"
                f"tags    : {len(flat)} over threshold "
                f"(general {threshold:.2f}, character {character_threshold:.2f})\n"
                f"timing  : load/cache {t_load * 1000:.0f} ms, "
                f"inference {t_infer * 1000:.0f} ms")
        log(info.replace("\n", " | "))

        return (tags_str, scored_str, prompt_string, info)
