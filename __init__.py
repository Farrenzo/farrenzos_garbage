"""
Farrenzo's Garbage Nodes
"""

import os
import toml
import json
import folder_paths

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
    },
    "GOOGLE_SIGLIP2": {
        "hf_repo":"google/siglip2-base-patch16-512",
        "directory": "siglip2"
    },
    "RESOURCE_LIBRARY_DIR": None
}


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
GOOGLE_SIGLIP2_INFO = NODE_SETTINGS["GOOGLE_SIGLIP2"]
GOOGLE_SIGLIP2_DIR  = os.path.join(current_path, "models", GOOGLE_SIGLIP2_INFO["directory"])
# Folder holding the resource library database and its .previews. Defaults to ComfyUI's own models folder.
RESOURCE_LIBRARY_DIR = NODE_SETTINGS.get("RESOURCE_LIBRARY_DIR") or folder_paths.models_dir


# Instantiate
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, log, install_xpu_patch, start_keepalive
with open(toml_path, "r") as f:
    VERSION = toml.load(f)["project"]["version"]

install_xpu_patch()
start_keepalive()
log(f"v{VERSION} has loaded {len(NODE_DISPLAY_NAME_MAPPINGS)} nodes.", "finish")
for _, n_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"    \033[0;37m {n_name} \033[0m")


from .nodes.fg_assistant import routes as _routes
WEB_DIRECTORY = "./web"
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY"
]
