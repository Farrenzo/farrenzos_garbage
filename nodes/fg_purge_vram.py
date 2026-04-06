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
                "anything": ("*", {}),
                "purge_cache": ("BOOLEAN", {"default": True}),
                "purge_models": ("BOOLEAN", {"default": True}),
            },
            "optional": {}
        }


    RETURN_TYPES = ()
    FUNCTION = "purge_vram"
    CATEGORY = "Farrenzo's Garbage/Utils"
    OUTPUT_NODE = True

    def purge_vram(self, anything, purge_cache, purge_models):
        clear_memory(purge_cache, purge_models)
        log(f"{self.NODE_NAME}: VRAM Cleared.")
        return ()

