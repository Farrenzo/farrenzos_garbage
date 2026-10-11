"""ComfyUI nodes for Depth Anything 3.
Model capability matrix:

Variant               head_type  has_sky  has_conf  cam_dec
DA3-Small             dualdpt    False    True      yes
DA3-Base              dualdpt    False    True      yes
DA3-Mono-Large        dpt        True     False     no
DA3-Metric-Large      dpt        True     False     no  (raw output is metres)
"""

from __future__ import annotations

import logging
import torch

import comfy.model_management as mm
import comfy.sd
import folder_paths
from comfy.ldm.depth_anything_3 import preprocess as da3_preprocess
from comfy_api.latest import io

DA3ModelType = io.Custom("DA3_MODEL")
DA3Geometry  = io.Custom("DA3_GEOMETRY")

# DA3_GEOMETRY is a dict with these optional keys (absent when the upstream model didn't produce them):
#
# Per-frame tensors - B = batch size in mono mode; B = S (number of views) in multi-view mode.
#   "depth":       torch.Tensor (B, H, W)         -- raw model depth (always present; matches MoGe convention)
#   "image":       torch.Tensor (B, H, W, 3)      -- source image in [0, 1], CPU (always present)
#   "mode":        str                            -- "mono" or "multiview" (always present)
#   "sky":         torch.Tensor (B, H, W)         -- sky probability in [0, 1] (Mono/Metric variants only)
#   "confidence":  torch.Tensor (B, H, W)         -- raw model confidence output (Small/Base variants only)
#
# Multi-view only - S = number of views; the leading 1 is the scene dimension from the model.
#   "extrinsics":  torch.Tensor (1, S, 3, 4)      -- world-to-camera [R|t] matrices
#   "intrinsics":  torch.Tensor (1, S, 3, 3)      -- pixel-space intrinsics
#
# DA3_POINT_CLOUD is a dict:
#   "points":     torch.Tensor (N, 3)  -- 3-D coords in glTF convention (Y-up, Z-back)
#   "colors":     torch.Tensor (N, 3)  -- RGB in [0, 1], or None
#   "confidence": torch.Tensor (N,)    -- raw confidence per point, or None


def _da3_unproject(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Pixel-space K⁻¹ unprojection: (H,W) depth → (H,W,3) point map in OpenCV space."""
    H, W = depth.shape
    u = torch.arange(W, dtype=torch.float32, device=depth.device)
    v = torch.arange(H, dtype=torch.float32, device=depth.device)
    u, v = torch.meshgrid(u, v, indexing='xy')             # both (H, W)
    pix = torch.stack([u, v, torch.ones_like(u)], dim=-1)  # (H, W, 3)
    rays = torch.einsum('ij,hwj->hwi', torch.linalg.inv(K.to(depth.device)), pix)
    return rays * depth.unsqueeze(-1)                       # (H, W, 3)


def _da3_default_K(H: int, W: int) -> torch.Tensor:
    """Fallback ~60° FOV pinhole K for mono-mode DA3 (no intrinsics in geometry)."""
    fx = fy = float(W) * 0.7
    return torch.tensor([[fx, 0.0, (W - 1) / 2.0],
                         [0.0, fy, (H - 1) / 2.0],
                         [0.0, 0.0, 1.0]], dtype=torch.float32)


def _da3_get_K(geometry: dict, b: int, H: int, W: int) -> torch.Tensor:
    """Return pixel-space K for batch element b, falling back to a default estimate."""
    if "intrinsics" in geometry:
        # shape (1, S, 3, 3) - leading scene dimension from the multiview head
        return geometry["intrinsics"][0, b].float()
    logging.getLogger("comfy").warning(
        "DA3_GEOMETRY has no intrinsics (mono-mode model). "
        "Using a ~60° FOV estimate; 3-D reconstruction may be inaccurate."
    )
    return _da3_default_K(H, W)


def _da3_get_extrinsic(geometry: dict, b: int) -> torch.Tensor | None:
    """Return the world-to-camera extrinsic for batch element b, or None in mono mode.

    The model outputs (1, S, 3, 4) [R|t] matrices; the fallback identity is (4, 4).
    _da3_apply_extrinsic handles both shapes via [:3, :3] / [:3, 3] slicing.
    """
    if "extrinsics" not in geometry:
        return None
    return geometry["extrinsics"][0, b].float()


def _da3_apply_extrinsic(points_cam: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Transform (H,W,3) OpenCV camera-space points to world space."""
    E = E.to(points_cam.device).float()
    if not torch.isfinite(E).all():
        logging.getLogger("comfy").warning(
            "DA3 extrinsic matrix contains non-finite values (pose estimation may have failed). "
            "Falling back to camera-space coordinates."
        )
        return points_cam
    H, W, _ = points_cam.shape
    R = E[:3, :3]           # (3, 3) rotation
    t = E[:3, 3]            # (3,)   translation
    R_inv = R.T             # rotation inverse = transpose for orthogonal R
    t_inv = -(R_inv @ t)    # (3,)
    pts = points_cam.reshape(-1, 3)                 # (N, 3)
    pts_world = pts @ R_inv.T + t_inv               # (N, 3)
    return pts_world.reshape(H, W, 3)


def _normalize_confidence(conf: torch.Tensor) -> torch.Tensor:
    """Map raw confidence to [0, 1] per image."""
    B = conf.shape[0]
    out = []
    for i in range(B):
        c = conf[i]
        c_min, c_max = c.min(), c.max()
        out.append((c - c_min) / (c_max - c_min) if c_max > c_min else torch.ones_like(c))
    return torch.stack(out, dim=0)


def _da3_build_mask(geometry: dict, b: int, H: int, W: int, confidence_threshold: float, use_sky_mask: bool) -> torch.Tensor:
    """Build (H,W) bool keep-mask from sky probability and confidence."""
    mask = torch.ones(H, W, dtype=torch.bool)
    if use_sky_mask and "sky" in geometry:
        mask = mask & (geometry["sky"][b] < 0.5)
    if "confidence" in geometry and confidence_threshold > 0.0:
        conf_norm = _normalize_confidence(geometry["confidence"][b:b + 1])[0]
        mask = mask & (conf_norm >= confidence_threshold)
    return mask


def _run_da3(model_patcher, image: torch.Tensor, process_res: int, method: str = "upper_bound_resize"):
    """Run DA3 on (B,H,W,3), returns depth/conf/sky at original resolution (or None)."""
    assert image.ndim == 4 and image.shape[-1] == 3, f"expected (B,H,W,3) IMAGE; got {tuple(image.shape)}"

    B, H, W, _ = image.shape
    mm.load_model_gpu(model_patcher)
    diffusion = model_patcher.model.diffusion_model
    device = mm.get_torch_device()
    dtype = diffusion.dtype if diffusion.dtype is not None else torch.float32

    depths, confs, skies = [], [], []
    for i in range(B):
        single = image[i:i + 1].to(device)
        x = da3_preprocess.preprocess_image(single, process_res=process_res, method=method)
        x = x.to(dtype=dtype)
        with torch.no_grad():
            out = diffusion(x)

        depth_lr = out["depth"]
        depth_full = torch.nn.functional.interpolate(
            depth_lr.unsqueeze(1).float(), size=(H, W),
            mode="bilinear", align_corners=False,
        ).squeeze(1).cpu()
        depths.append(depth_full)

        if "depth_conf" in out:
            conf_full = torch.nn.functional.interpolate(
                out["depth_conf"].unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu()
            confs.append(conf_full)
        if "sky" in out:
            sky_full = torch.nn.functional.interpolate(
                out["sky"].unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu()
            skies.append(sky_full)

    depth = torch.cat(depths, dim=0)
    confidence = torch.cat(confs, dim=0) if confs else None
    sky = torch.cat(skies, dim=0) if skies else None
    return depth, confidence, sky


class FG_DepthAnything_V3Inferer:

    @classmethod
    def _load_model(cls, model_name, weight_dtype):
        model_options = {}
        if weight_dtype == "fp16":
            model_options["dtype"] = torch.float16
        elif weight_dtype == "bf16":
            model_options["dtype"] = torch.bfloat16
        elif weight_dtype == "fp32":
            model_options["dtype"] = torch.float32

        path = folder_paths.get_full_path_or_raise("geometry_estimation", model_name)
        model = comfy.sd.load_diffusion_model(path, model_options=model_options)
        return model

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required":{
                "image"        : ("IMAGE", ),
                "model_name"   : (folder_paths.get_filename_list("geometry_estimation"), ),
                "weight_dtype" : (["default", "fp16", "bf16", "fp32"], {"default":"default"} ), 
                "resolution"   : (
                    "INT", 
                    {
                        "default": 504, "min": 140, "max": 2520, "step": 14,
                        "tooltip": "Resolution the model runs at (longest side, multiple of 14). Lower = faster / less VRAM. Higher = more detail. Output is upsampled back to the original size."
                    }
                ),
                "resize_method": (
                    ["upper_bound_resize", "lower_bound_resize"],
                    {
                        "default":"upper_bound_resize",
                        "tooltip": "upper_bound_resize: scale so the longest side = resolution (caps memory, default). lower_bound_resize: scale so the shortest side = resolution (preserves more detail on tall/wide images, uses more memory)."
                    }
                ), 
                "mode"         : (
                    ["mono", "multiview"],
                    {
                        "default":"mono",
                        "tooltip": "mono: single view image (works with any model variant). multiview: all images processed together for geometric consistency + camera pose (for Small/Base models only)."
                    }
                ),
                "ref_view_strategy": (
                    ["saddle_balanced", "saddle_sim_range", "first", "middle"],
                    {
                        "default": "saddle_balanced",
                        "tooltip": "Which view acts as the geometric anchor. - saddle_balanced: the view most 'average' across all others (best general choice). - saddle_sim_range: the view most visually distinct from the others. - first / middle: fixed positional picks."
                    }
                ),
                "pose_method": (
                    ["cam_dec", "ray_pose"],
                    {
                        "default": "cam_dec",
                        "tooltip": "How the camera field-of-view is estimated (for Small/Base models only). - cam_dec: learned from image features. - ray_pose: derived geometrically from the model's 3D ray output. Affects perspective correctness of the 3D output. Try both if results look distorted."
                    }
                ),

            }
        }
    

    CATEGORY = "Farrenzo's Garbage/Image/Geometry Estimation",
    RETURN_TYPES = ("DA3_GEOMETRY", )
    RETURN_NAMES = ("da3_geometry", )
    DISPLAY_NAME = "🗑️ Depth Anything",
    SEARCH_ALIASES  = ["depth", "geometry", "da3", "depth anything", "monocular", "pointmap", "sky", "3d", "metric depth", "disparity"]
    OUTPUT_TOOLTIPS = ("Dictionary of non-normalized tensors. Always has the keys: depth, image, mode. Optional keys: sky (for Mono/Metric), confidence (for Small/Base), extrinsics + intrinsics (for multi-view).",)

    @classmethod
    def execute(cls, model_name, weight_dtype, image, resolution, resize_method, mode) -> io.NodeOutput:
        da3_model = cls._load_model(model_name, weight_dtype)
        mode_val = mode["mode"]  # "mono" or "multiview"

        if mode_val == "mono":
            return cls._execute_mono(da3_model, image, resolution, resize_method)

        # Capability checks for multi-view mode.
        diffusion = da3_model.model.diffusion_model
        pose_method = mode["pose_method"]
        ref_view_strategy = mode["ref_view_strategy"]

        has_cam_dec = diffusion.cam_dec is not None
        has_dualdpt = diffusion.head_type == "dualdpt"

        if not has_cam_dec and not has_dualdpt:
            raise ValueError(
                "multi-view mode requires Small or Base model. The loaded model "
                f"(head_type='{diffusion.head_type}') does not support cross-view "
                "attention or camera pose estimation. Switch mode to 'mono', or "
                "load Small or Base model for mult-view."
            )

        if pose_method == "cam_dec" and not has_cam_dec:
            raise ValueError(
                "pose_method='cam_dec' requires a camera decoder, but the loaded "
                f"model (head_type='{diffusion.head_type}') does not have one. "
                "Use pose_method='ray_pose' instead."
            )
        if pose_method == "ray_pose" and not has_dualdpt:
            raise ValueError(
                "pose_method='ray_pose' requires a DualDPT head, but the loaded "
                f"model has a '{diffusion.head_type}' head. "
                "Use pose_method='cam_dec' instead."
            )

        return cls._execute_multiview(
            da3_model, image, resolution, resize_method,
            ref_view_strategy, pose_method,
        )

    @classmethod
    def _execute_mono(cls, model, image, resolution, resize_method) -> io.NodeOutput:
        depth, confidence, sky = _run_da3(model, image, resolution, method=resize_method)

        geometry: dict = {
            "depth": depth.contiguous(),
            "image": image[..., :3].cpu(),
            "mode": "mono",
        }
        if sky is not None:
            geometry["sky"] = sky.contiguous()
        if confidence is not None:
            geometry["confidence"] = confidence.contiguous()
        return io.NodeOutput(geometry)

    @classmethod
    def _execute_multiview(cls, model, image, resolution, resize_method, ref_view_strategy, pose_method) -> io.NodeOutput:
        assert image.ndim == 4 and image.shape[-1] == 3, \
            f"expected (B,H,W,3) IMAGE; got {tuple(image.shape)}"
        S, H, W, _ = image.shape

        mm.load_model_gpu(model)
        diffusion = model.model.diffusion_model
        device = mm.get_torch_device()
        dtype = diffusion.dtype if diffusion.dtype is not None else torch.float32

        # All views in a single forward pass: (1, S, 3, H', W').
        x = image.to(device)
        x = da3_preprocess.preprocess_image(x, process_res=resolution, method=resize_method)
        x = x.to(dtype=dtype).unsqueeze(0)

        use_ray_pose = (pose_method == "ray_pose")
        with torch.no_grad():
            out = diffusion(x, use_ray_pose=use_ray_pose, ref_view_strategy=ref_view_strategy)

        depth = torch.nn.functional.interpolate(
            out["depth"].float().unsqueeze(1), size=(H, W),
            mode="bilinear", align_corners=False,
        ).squeeze(1).cpu()

        sky = None
        if "sky" in out:
            sky = torch.nn.functional.interpolate(
                out["sky"].unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu()

        if "extrinsics" in out and "intrinsics" in out:
            extrinsics = out["extrinsics"].float().cpu()
            intrinsics = out["intrinsics"].float().cpu()
        else:
            extrinsics = torch.eye(4)[None, None].expand(1, S, 4, 4).clone()
            intrinsics = torch.eye(3)[None, None].expand(1, S, 3, 3).clone()

        geometry: dict = {
            "depth": depth.contiguous(),
            "image": image[..., :3].cpu(),
            "mode": "multiview",
            "extrinsics": extrinsics.contiguous(),
            "intrinsics": intrinsics.contiguous(),
        }
        if sky is not None:
            geometry["sky"] = sky.contiguous()
        if "depth_conf" in out:
            conf = torch.nn.functional.interpolate(
                out["depth_conf"].unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu()
            geometry["confidence"] = conf.contiguous()
        return io.NodeOutput(geometry)

