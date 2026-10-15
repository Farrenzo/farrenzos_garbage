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

class FG_MinimumMaximum:
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

    def get_min_or_max(self, mode, int_1, int_2, multiplier=1):
        if mode:
            return (multiplier*max(int_1, int_2),)
        else:
            return (multiplier*min(int_1, int_2),)

"""
┌─────────────────────────────────────┐
│            Int / Float Out          │
├─────────────────────────────────────┤
│                         Integer  ○  │
│                           Float  ○  │
│                            Bool  ○  │
│                                     │
│ <→ SWITCH> Return True/False        │
│ <→ INPUT> True Int                  │
│ <→ INPUT> True Float                │
│ <→ INPUT> False Int                 │
│ <→ INPUT> False Float               │
│                                     │
└─────────────────────────────────────┘
"""
class FG_IntFloatOut:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": ("BOOLEAN", {"default": True}),
                "true_int": ("INT", {"default": 0}),
                "true_float": ("FLOAT", {"default": 0.0}),
                "false_int": ("INT", {"default": 0}),
                "false_float": ("FLOAT", {"default": 0.0}),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "BOOLEAN", )
    RETURN_NAMES = ("Integer", "Float", "Boolean")
    FUNCTION = "execute"
    CATEGORY = "utils"

    def execute(
        self,
        switch,
        true_int,
        true_float,
        false_int,
        false_float,
    ):
        if switch:
            return (true_int, true_float, switch)

        return (false_int, false_float, switch)
