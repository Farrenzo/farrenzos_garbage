"""
LAB Color Transfer - ComfyUI Custom Node
Recolors TARGET to match the color statistics of REFERENCE.
"""

import cv2
import torch
import numpy as np


def tensor_to_numpy_bgr(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.squeeze(0).cpu().numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def numpy_bgr_to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
    return tensor.unsqueeze(0)


def lab_color_transfer(
    target_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    strength: float = 1.0,
    match_l: bool = True,
    match_a: bool = True,
    match_b: bool = True
) -> np.ndarray:
    """Recolor target_bgr to match the LAB statistics of reference_bgr."""
    tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    ref_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

    result_lab = tgt_lab.copy()
    channels = {0: match_l, 1: match_a, 2: match_b}

    for ch, do_transfer in channels.items():
        if not do_transfer:
            continue
        tgt_mean = tgt_lab[:, :, ch].mean()
        tgt_std  = tgt_lab[:, :, ch].std()
        ref_mean = ref_lab[:, :, ch].mean()
        ref_std  = ref_lab[:, :, ch].std()

        if tgt_std < 1e-6:
            continue

        transferred = (tgt_lab[:, :, ch] - tgt_mean) / tgt_std * ref_std + ref_mean
        result_lab[:, :, ch] = tgt_lab[:, :, ch] * (1.0 - strength) + transferred * strength

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


class FG_LABColorTransfer:
    def __init__(self):
        self.NODE_NAME = "LAB Color Transfer"

    CATEGORY = "Farrenzo's Garbage/Image"
    FUNCTION = "transfer"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    DESCRIPTION = "Recolors TARGET to match the color statistics of REFERENCE."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("IMAGE", {"tooltip": "Image to recolor"}),
                "reference": ("IMAGE", {"tooltip": "Image to copy colors from"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "0 = no change, 1 = full transfer, >1 = over-correct"}),
            },
            "optional": {
                "match_luminance": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Transfer the L (brightness/luminance) channel"
                }),
                "match_green_red": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Transfer the A (green-red) channel"
                }),
                "match_blue_yellow": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Transfer the B (blue-yellow) channel"
                }),
                "mask": ("MASK", {
                    "tooltip": "Optional mask (1 = apply transfer, 0 = keep original)."
                }),
            },
        }

    def transfer(self,
                 target: torch.Tensor,
                 reference: torch.Tensor,
                 strength: float = 1.0,
                 match_luminance: bool = True,
                 match_green_red: bool = True,
                 match_blue_yellow: bool = True,
                 mask: torch.Tensor | None = None) -> tuple:

        batch_size = target.shape[0]
        results = []

        if reference.shape[0] < batch_size:
            reference = reference.expand(batch_size, -1, -1, -1)

        for i in range(batch_size):
            tgt_bgr = tensor_to_numpy_bgr(target[i].unsqueeze(0))
            ref_bgr = tensor_to_numpy_bgr(reference[i].unsqueeze(0))

            if tgt_bgr.shape[:2] != ref_bgr.shape[:2]:
                ref_bgr = cv2.resize(ref_bgr, (tgt_bgr.shape[1], tgt_bgr.shape[0]),
                                     interpolation=cv2.INTER_AREA)

            transferred_bgr = lab_color_transfer(
                tgt_bgr, ref_bgr, strength,
                match_l=match_luminance,
                match_a=match_green_red,
                match_b=match_blue_yellow,
            )

            result_tensor = numpy_bgr_to_tensor(transferred_bgr)

            if mask is not None:
                m = mask[i] if mask.shape[0] > i else mask[0]
                m = m.unsqueeze(-1).cpu()
                if m.shape[:2] != result_tensor.shape[1:3]:
                    m_np = m.numpy()
                    m_np = cv2.resize(m_np, (result_tensor.shape[2], result_tensor.shape[1]),
                                      interpolation=cv2.INTER_LINEAR)
                    m = torch.from_numpy(m_np).unsqueeze(-1)

                original = target[i].cpu()
                result_tensor = result_tensor.squeeze(0)
                result_tensor = original * (1.0 - m) + result_tensor * m
                result_tensor = result_tensor.unsqueeze(0)

            results.append(result_tensor)

        output = torch.cat(results, dim=0)
        return (output,)


