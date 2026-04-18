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
tompl_path   = os.path.join(current_path, "pyproject.toml")

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


# LoRA index setup
def set_up_lora_index() -> dict:
    lora_index_file_path = folder_paths.get_full_path("loras", "fg_dynamic_lora_loader.json")
    lora_folder_paths = folder_paths.folder_names_and_paths["loras"][0]
    if not lora_index_file_path:
        lora_files = {}
        for folder_path in lora_folder_paths:
            for walk_path, _, file_names in os.walk(folder_path):
                for file_name in file_names:
                    if file_name.endswith(".safetensors"):
                        rel_path = os.path.relpath(os.path.join(walk_path, file_name), folder_path)
                        lora_files[rel_path] = {"trigger_words": "", "preview_image": ""}
        with open(f"{lora_folder_paths[0]}/fg_dynamic_lora_loader.json", "a+") as lora_index_file:
            lora_index_file.write(json.dumps(lora_files))
        return lora_files

    with open(lora_index_file_path, "r") as lora_index_file:
        lora_files: dict = json.loads(lora_index_file.read())
    return lora_files

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


# LoRRA Loader section.
_lora_cache = None
_lora_mtime = 0

def _get_lora_index_path():
    return folder_paths.get_full_path("loras", "fg_dynamic_lora_loader.json")

# Still run at startup to create the file if missing
set_up_lora_index()

@PromptServer.instance.routes.get("/fg/lora_index")
async def get_lora_index(request):
    global _lora_cache, _lora_mtime
    index_path = _get_lora_index_path()
    if index_path:
        current_mtime = os.path.getmtime(index_path)
        if _lora_cache is None or current_mtime != _lora_mtime:
            with open(index_path, "r") as f:
                _lora_cache = json.load(f)
            _lora_mtime = current_mtime
    return web.json_response(_lora_cache or {})

# Instantiate
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, log
with open(tompl_path, "r") as f:
    VERSION = toml.load(f)["project"]["version"]

log(f"v{VERSION} has loaded {len(NODE_DISPLAY_NAME_MAPPINGS)} nodes.", "finish")
for _, n_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"    \033[0;37m {n_name} \033[0m")

WEB_DIRECTORY = "./web"
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY"
]
