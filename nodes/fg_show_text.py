import json
from ._fg_helperfunctions import log

class ShowText:
    """Displays text in the node after execution. Text persists across tab switches."""
    def __init__(self):
        self.NODE_NAME = "Show Text"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "show"
    CATEGORY = "Farrenzo's Garbage/Utils"
    OUTPUT_NODE = True

    def show(self, text, unique_id=None, extra_pnginfo=None):
        # Build values list
        values = []
        for val in text:
            try:
                if isinstance(val, str):
                    values.append(val)
                elif isinstance(val, list):
                    values.extend(val)
                else:
                    values.append(json.dumps(val))
            except Exception:
                values.append(str(val))
        
        # Persist to workflow (survives tab switches)
        if extra_pnginfo is not None:
            if isinstance(extra_pnginfo, list) and len(extra_pnginfo) > 0:
                if isinstance(extra_pnginfo[0], dict) and "workflow" in extra_pnginfo[0]:
                    workflow = extra_pnginfo[0]["workflow"]
                    node = next(
                        (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                        None
                    )
                    if node:
                        node["widgets_values"] = [values]  # Wrap in list like ShowAnything
        
        # Debug output
        display = "\n".join(values)
        log(f"{self.NODE_NAME}:\n{'='*50}\n{display}\n{'='*50}\n")
        
        # Return format matching ShowAnything
        if len(values) == 1:
            return {"ui": {"text": values}, "result": (values[0],)}
        else:
            return {"ui": {"text": values}, "result": (values,)}