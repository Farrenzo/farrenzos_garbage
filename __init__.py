"""
Farrenzo's Garbage Nodes
"""

import os
import toml
import json
import folder_paths
from aiohttp import web
from server import PromptServer

# Paths
current_path = os.path.dirname(__file__)
env_path     = os.path.join(current_path, "_env.json")
toml_path    = os.path.join(current_path, "pyproject.toml")


initial_settings = {
    "TELEGRAM_CHAT_ID": None,
    "TELEGRAM_PRIVATE_API": None,
    "WD_14_TAGGER": {
        "directory": "wd14_v3",
        "tagging_models":{
            "eva02-large": {
                "model": "model.onnx",
                "csv": "wd-eva02-large-tagger-v3.csv"
            }
        }
    }
}


PREVIEW_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
# LoRA index setup
def set_up_lora_index() -> tuple[dict, str, str]:
    lora_folder_paths = folder_paths.folder_names_and_paths["loras"][0]
    loras = folder_paths.get_folder_paths("loras")
    setup = True
    for lora_paths in loras:
        if os.path.isdir(f"{lora_paths}/.lora_previews"):
            setup = False
            lora_previews_path = f"{lora_paths}/.lora_previews"
            break

    if setup:
        lora_previews_path = f"{loras[0]}/.lora_previews"
        print(f"Created `.lora_previews` folder at {lora_previews_path}")
        os.makedirs(lora_previews_path)

    lora_index_file_path = folder_paths.get_full_path("loras", "/.lora_previews/fg_dynamic_lora_loader.json")
    if not lora_index_file_path:
        lora_index_file_path = f"{lora_folder_paths[0]}/.lora_previews/fg_dynamic_lora_loader.json"
        lora_files = {}
        for folder_path in lora_folder_paths:
            for walk_path, _, file_names in os.walk(folder_path):
                for file_name in file_names:
                    if file_name.endswith(".safetensors"):
                        rel_path = os.path.relpath(os.path.join(walk_path, file_name), folder_path)
                        lora_files[rel_path] = {"trigger_words": ""}
        with open(lora_index_file_path, "a+") as lora_index_file:
            lora_index_file.write(json.dumps(lora_files))
        return lora_files, lora_index_file_path, lora_previews_path

    with open(lora_index_file_path, "r") as lora_index_file:
        lora_files: dict = json.loads(lora_index_file.read())
    return lora_files, lora_index_file_path, lora_previews_path

# Telegram setup
if not os.path.isfile(env_path):
    with open(env_path, "a+") as env_file:
        env_file.write(json.dumps(initial_settings, indent=4))
        NODE_SETTINGS = initial_settings
else:
    with open(env_path, "r", encoding="utf-8") as settings_file:
        NODE_SETTINGS = json.loads(settings_file.read())


TELEGRAM_CHAT_ID = NODE_SETTINGS["TELEGRAM_CHAT_ID"]
TELEGRAM_PRIVATE_API = NODE_SETTINGS["TELEGRAM_PRIVATE_API"]
WD_14_INFO = NODE_SETTINGS["WD_14_TAGGER"]
model_path = os.path.join(current_path, f"models/{WD_14_INFO['directory']}")
WD_14_INFO["model_path"] = model_path


# LoRA Loader section.
_lora_cache = None
_lora_mtime = 0
# Still run at startup to create the file if missing
lora_files, lora_index_file_path, lora_previews_path = set_up_lora_index()

@PromptServer.instance.routes.get("/fg/lora_index")
async def get_lora_index(request):
    global _lora_cache, _lora_mtime
    if lora_index_file_path:
        current_mtime = os.path.getmtime(lora_index_file_path)
        if _lora_cache is None or current_mtime != _lora_mtime:
            with open(lora_index_file_path, "r") as f:
                _lora_cache = json.load(f)
            _lora_mtime = current_mtime
    return web.json_response(_lora_cache or {})

@PromptServer.instance.routes.get("/fg/lora_previews")
async def get_lora_previews(_request):
    """Returns { stem: filename } for every image in .lora_previews/."""
    mapping = {}
    if os.path.isdir(lora_previews_path):
        for fname in os.listdir(lora_previews_path):
            stem, ext = os.path.splitext(fname)
            if ext.lower() in PREVIEW_EXTS:
                mapping[stem] = fname
    return web.json_response(mapping)

@PromptServer.instance.routes.get("/fg/lora_preview/{filename}")
async def get_lora_preview(request):
    """Serves a single preview image from .lora_previews/."""
    filename = request.match_info["filename"]
    # Guard against path traversal — only allow plain filenames.
    if "/" in filename or "\\" in filename or filename.startswith("."):
        return web.json_response({"error": "Invalid filename"}, status=400)
    path = os.path.join(lora_previews_path, filename)
    if not os.path.isfile(path):
        return web.json_response({"error": "Not found"}, status=404)
    return web.FileResponse(path)

# Instantiate
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, log, _install_patch
with open(toml_path, "r") as f:
    VERSION = toml.load(f)["project"]["version"]

_install_patch()
log(f"v{VERSION} has loaded {len(NODE_DISPLAY_NAME_MAPPINGS)} nodes.", "finish")
for _, n_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"    \033[0;37m {n_name} \033[0m")

WEB_DIRECTORY = "./web"
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY"
]
