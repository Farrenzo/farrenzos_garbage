"""
Returns the Loads an image, outputs the image, image mask, width & height.

┌─────────────────────────────────────┐
│ Minimum vs Maximum                  │
├─────────────────────────────────────┤
│ ○ int_1                 Integer  ○  |
│ ○ int_2                             |
│                                     │
│ <SWITCH mode default TRUE>          │
└─────────────────────────────────────┘

"""

class MinimumMaximum:
    def __init__(self):
        self.NODE_NAME = "Minimum + Maximum"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": ("BOOLEAN", {"default": True, "label_on": "max", "label_off": "min"}),
                "int_1": ("INT",),
                "int_2": ("INT",),
            },
        }

    FUNCTION = "get_min_or_max"
    CATEGORY = "Farrenzo's Garbage/Utils"
    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("Integer", )

    def get_min_or_max(self, mode, int_1, int_2):
        if mode:
            return (max(int_1, int_2),)
        else:
            return (min(int_1, int_2),)

