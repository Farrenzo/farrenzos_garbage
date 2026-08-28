import math


class FG_SeedVR2CalculatorNode:
    """Calculates SeedVR2 short-edge settings, target output dimensions,

    total frame counts, and estimated rendering times.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_width": ("INT", {"default": 480, "min": 1, "max": 8192}),
                "input_height": ("INT", {"default": 856, "min": 1, "max": 8192}),
                "target_long_edge": (
                    "INT",
                    {"default": 1280, "min": 1, "max": 8192},
                ),
                "duration_seconds": (
                    "FLOAT",
                    {"default": 30.0, "min": 0.1, "max": 86400.0},
                ),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0}),
                "sec_per_frame": (
                    "FLOAT",
                    {"default": 2.5, "min": 0.01, "max": 120.0},
                ),
                "multiple_alignment": ([8, 16], {"default": 16}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "seedvr2_resolution",
        "output_width",
        "output_height",
        "total_frames",
        "eta_formatted",
    )
    FUNCTION = "calculate"
    CATEGORY = "SeedVR2/Utils"

    def calculate(
        self,
        input_width: int,
        input_height: int,
        target_long_edge: int,
        duration_seconds: float,
        fps: float,
        sec_per_frame: float,
        multiple_alignment: int = 16,
    ):
        # Determine orientation and dimensions
        short_edge = min(input_width, input_height)
        long_edge = max(input_width, input_height)
        is_portrait = input_height > input_width

        # Calculate exact SeedVR2 short-edge target parameter
        scale_ratio = target_long_edge / long_edge
        calculated_short_edge = int(round(short_edge * scale_ratio))

        # Align to model/codec requirement (divisible by 8 or 16)
        aligned_short_edge = (
            math.ceil(calculated_short_edge / multiple_alignment)
            * multiple_alignment
        )
        aligned_long_edge = (
            math.ceil(target_long_edge / multiple_alignment)
            * multiple_alignment
        )

        if is_portrait:
            output_w = aligned_short_edge
            output_h = aligned_long_edge
        else:
            output_w = aligned_long_edge
            output_h = aligned_short_edge

        # Frame and time estimations
        total_frames = int(round(duration_seconds * fps))
        total_seconds = total_frames * sec_per_frame

        hours, remainder = divmod(int(total_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            eta_formatted = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            eta_formatted = f"{minutes}m {seconds}s"
        else:
            eta_formatted = f"{seconds}s"

        print("\n--- SeedVR2 Pre-Calculation ---")
        print(f"Input Resolution : {input_width}x{input_height}")
        print(f"SeedVR2 Setting  : {aligned_short_edge} (resolution param)")
        print(f"Target Output    : {output_w}x{output_h}")
        print(f"Total Frames     : {total_frames}")
        print(f"Estimated Time   : {eta_formatted}")
        print("-------------------------------\n")

        return (
            aligned_short_edge,
            output_w,
            output_h,
            total_frames,
            eta_formatted,
        )


# Example usage as a standalone script:
if __name__ == "__main__":
    calc = FG_SeedVR2CalculatorNode()
    seed_res, out_w, out_h, frames, eta = calc.calculate(
        input_width=480,
        input_height=856,
        target_long_edge=1280,
        duration_seconds=10.0,
        fps=30.0,
        sec_per_frame=2.0,
        multiple_alignment=16,
    )