"""
"""
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

