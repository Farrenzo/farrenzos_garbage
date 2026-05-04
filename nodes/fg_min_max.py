"""
Returns the Loads an image, outputs the image, image mask, width & height.

┌─────────────────────────────────────┐
│ Minimum vs Maximum                  │
├─────────────────────────────────────┤
│ ○ int_1                 Integer  ○  |
│ ○ int_2                             |
│                                     │
│ <SWITCH mode default TRUE>          │
│ <INT Multiplier>                    │
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
                "int_1": ("INT", {"default": 0, "forceInput": True}),
                "int_2": ("INT", {"default": 0, "forceInput": True}),
            },
            "optional": {
                "multiplier": ("INT", {"tooltip": "Multiply the output by a given value."}),
            },
        }

    FUNCTION = "get_min_or_max"
    CATEGORY = "Farrenzo's Garbage/Utils"
    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("Integer", )

    def get_min_or_max(self, mode, int_1, int_2, int_3=1):
        if mode:
            return (1*max(int_1, int_2),)
        else:
            return (1*min(int_1, int_2),)

