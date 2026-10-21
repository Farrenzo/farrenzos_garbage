"""
"""
import os
import sys
import time
import threading
import subprocess

from aiohttp import web
from server import PromptServer

from ._fg_helperfunctions import log, clear_memory

class _AnyType(str):
    """Matches any ComfyUI socket type for a passthrough utility node."""
    def __ne__(self, other):
        return False


_ANY = _AnyType("*")


class FG_PurgeMemory:
    """
    Drop this anywhere in a graph as an explicit 'empty the card' step.
    Passes its `data` input through unchanged, so it can sit inline between
    any two nodes without breaking the graph — e.g. between a VAE decode
    and the next model load, to guarantee the decoder is actually gone
    before the next stage tries to claim VRAM.
    """
    def __init__(self):
        self.NODE_NAME = "Purge Memory"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "purge_models": ("BOOLEAN", {"default": True}),
                "nuclear": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "anything": (_ANY,),
            }
        }

    RETURN_TYPES = (_ANY,)
    RETURN_NAMES = ("anything",)
    FUNCTION = "purge_vram"
    CATEGORY = "Farrenzo's Garbage/Utils"
    DESCRIPTION = "Terminal node. Clears all cache."
    OUTPUT_NODE = True

    def purge_vram(self, anything=None, purge_models=False, nuclear=False):
        clear_memory(purge_cache=True, purge_models=purge_models, nuclear=nuclear)
        log(f"{self.NODE_NAME}: VRAM Cleared.")
        return (anything,)


"""
POST /fg/restart  ->  relaunches ComfyUI with the same interpreter, flags and args.
Imported once from the node pack's __init__.py; registers the route on import.
"""

def _build_cmd():
    # Interpreter flags like -s (portable builds) aren't in sys.argv, so recover them.
    cmd = [sys.executable, *subprocess._args_from_interpreter_flags(), *sys.argv]
    # Don't open a second browser tab; the existing tab reloads itself.
    if "--disable-auto-launch" not in cmd:
        cmd.append("--disable-auto-launch")
    return cmd


def _restart():
    time.sleep(0.75)  # let the HTTP response reach the browser first
    cmd = _build_cmd()
    log(f"Restart: relaunching -> {' '.join(cmd)}")
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        # os.execv on Windows doesn't quote args with spaces; spawn properly, then exit.
        subprocess.Popen(cmd, cwd=os.getcwd())
        os._exit(0)
    else:
        os.execv(sys.executable, cmd)


@PromptServer.instance.routes.post("/fg/restart")
async def fg_restart(request):
    threading.Thread(target=_restart, daemon=True).start()
    return web.json_response({"status": "restarting"})
