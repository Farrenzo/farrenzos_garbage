"""
"""
from ._fg_helperfunctions import log, clear_memory

class FG_PurgeMemory:

    def __init__(self):
        self.NODE_NAME = "Purge Memory"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "purge_cache": ("BOOLEAN", {"default": True}),
                "purge_models": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "anything": ("*", {}),
            }
        }


    RETURN_TYPES = ()
    FUNCTION = "purge_vram"
    CATEGORY = "Farrenzo's Garbage/Utils"
    DESCRIPTION = "Terminal node. Clears all cache."
    OUTPUT_NODE = True

    def purge_vram(self, anything=None, purge_cache=True, purge_models=True):
        clear_memory(purge_cache, purge_models)
        log(f"{self.NODE_NAME}: VRAM Cleared.")
        return ()

