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
                "multiplier": ("INT", {"default": 1, "tooltip": "Multiply the output by a given value."}),
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


"""
┌─────────────────────────────────────┐
│             Any Switch              │
├─────────────────────────────────────┤
│ ○ on_true_0            switch    ○  │
│ ○ on_false_0           MODEL_0   ○  │
│ ○ on_true_1            IMAGE_1   ○  │
│ ○ on_false_1           out_2     ○  │  <- spare pair, grows on connect
│ ○ on_true_2                         │
│ ○ on_false_2                        │
│ <SWITCH true/false>                 │
└─────────────────────────────────────┘
Pair N routes on_true_N or on_false_N to out_N. Only the selected side is
evaluated (lazy), so an unused LoRA loader / image load never runs.
"""


class AnyType(str):
    """Wildcard type: compares equal to every other type during validation."""
    def __ne__(self, other):
        return False


ANY = AnyType("*")
MAX_PAIRS = 16  # hard ceiling; the JS only shows what's in use + 1 spare


class FG_AnySwitch:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(MAX_PAIRS):
            optional[f"on_true_{i}"] = (ANY, {"lazy": True})
            optional[f"on_false_{i}"] = (ANY, {"lazy": True})
        return {
            "required": {
                "switch": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
            },
            "optional": optional,
        }

    # Output 0 is the switch itself; out_N lives at slot N + 1. The JS only
    # ever trims/grows from the END so indices stay aligned with this tuple.
    RETURN_TYPES = ("BOOLEAN",) + (ANY,) * MAX_PAIRS
    RETURN_NAMES = ("switch",) + tuple(f"out_{i}" for i in range(MAX_PAIRS))
    FUNCTION = "execute"
    CATEGORY = "Farrenzo's Garbage/Utils"

    @staticmethod
    def _prefix(switch):
        return "on_true_" if switch else "on_false_"

    def check_lazy_status(self, switch, **kwargs):
        # Linked-but-unevaluated lazy inputs arrive as None; unlinked ones are
        # absent entirely. Request only the selected side.
        prefix = self._prefix(switch)
        return [k for k, v in kwargs.items() if k.startswith(prefix) and v is None]

    def execute(self, switch, **kwargs):
        prefix = self._prefix(switch)
        # An unconnected selected side yields None for that out_N.
        return (switch,) + tuple(kwargs.get(f"{prefix}{i}") for i in range(MAX_PAIRS))


