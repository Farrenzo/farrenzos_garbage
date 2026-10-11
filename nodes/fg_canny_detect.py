"""
Canny Edges (Sequential) - ComfyUI Node
=======================================
Canny edge detection that never puts more than one small batch on the GPU.

Core's Canny node moves the ENTIRE image batch to the device in one go, so a
loaded video (hundreds/thousands of frames) floods VRAM and takes the system
down. This node works from disk instead:

  * source_path is a VIDEO  -> split into <output>/frames/ (PNG, grayscale),
                               then edges are run over that folder.
  * source_path is a FOLDER -> its images are used as-is (natural sort order).
  * source_path is an IMAGE -> just that one file.
  * source_path is empty    -> falls back to the optional IMAGE input, which
                               is chunked to the device batch_size at a time
                               (drop-in replacement for core Canny).

Edges are written to <output>/edges/ with the same base name as each source
frame. Only `batch_size` frames are ever on the device at once; each batch is
pulled back to CPU and written out before the next is loaded. With
skip_existing on, an interrupted run picks up where it stopped, and a video
that was already split is not split again.

┌─────────────────────────────────────┐
│ Canny Edges (Sequential)            │
├─────────────────────────────────────┤
│ ○ image (optional)          Edges ○ │
│                      Edges Folder ○ │
│                       Frame Count ○ │
│                               FPS ○ │
│ <→ TEXT  _ source_path>             │
│ - video file / frame folder / image │
│ <→ TEXT  _ output_folder>           │
│ - empty = output/fg_canny/<name>    │
│ <FLOAT _ low_threshold>             │
│ <FLOAT _ high_threshold>            │
│ <INT   _ batch_size>                │
│ <BOOL  _ skip_existing>             │
│ <BOOL  _ return_images>             │
└─────────────────────────────────────┘
"""

import os
import re
import json
import datetime

import av
import cv2
import torch
import numpy as np
from kornia.filters import canny

import folder_paths
import comfy.utils
import comfy.model_management as comfy_mm
from ._fg_helperfunctions import log


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif",
              ".mpg", ".mpeg", ".wmv", ".flv", ".ts", ".mts", ".m2ts")

EXTRACT_MARKER = ".fg_extracted.json"
FRAME_PATTERN = re.compile(r"^frame_\d{6}\.png$")


# --------------------------------------------------------------------------- #
# Disk helpers
# --------------------------------------------------------------------------- #
# cv2.imread / imwrite choke on non-ASCII paths on Windows, so everything goes
# through imdecode / imencode on raw bytes instead.

def _read_gray(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def _write_png(path: str, array: np.ndarray) -> None:
    # Compression 1: edge maps are mostly black and compress fine at the
    # fastest level; the default (3) just burns CPU time per frame.
    ok, buf = cv2.imencode(".png", array, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise IOError(f"Could not encode PNG: {path}")
    buf.tofile(path)


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _list_frames(folder: str) -> list:
    names = [
        n for n in os.listdir(folder)
        if not n.startswith(".")
        and n.lower().endswith(IMAGE_EXTS)
        and os.path.isfile(os.path.join(folder, n))
    ]
    names.sort(key=_natural_key)
    return [os.path.join(folder, n) for n in names]


def _clean_path_text(text: str) -> str:
    # Windows "Copy as path" wraps in quotes.
    return (text or "").strip().strip('"').strip("'").strip()


def _resolve_source(source_path: str):
    """Absolute path for the source, or None when nothing was given.
    Relative paths are tried against ComfyUI's input dir, then output dir."""
    source = _clean_path_text(source_path)
    if not source:
        return None
    if not os.path.isabs(source):
        for base in (folder_paths.get_input_directory(), folder_paths.get_output_directory()):
            candidate = os.path.join(base, source)
            if os.path.exists(candidate):
                return os.path.normpath(candidate)
    return os.path.normpath(source)


def _resolve_output_folder(output_folder: str, default_name: str) -> str:
    name = _clean_path_text(output_folder) or f"fg_canny/{default_name}"
    name = name.replace("%HMSf%", f"{datetime.datetime.now():%m%d%H%M%S%f}")
    out = name if os.path.isabs(name) else os.path.join(folder_paths.get_output_directory(), name)
    out = os.path.normpath(out)
    os.makedirs(out, exist_ok=True)
    return out


def _source_signature(path: str) -> dict:
    st = os.stat(path)
    return {"source": os.path.abspath(path), "size": st.st_size, "mtime": st.st_mtime}


def _extract_video(video_path: str, frames_dir: str) -> tuple:
    """Split a video into frames_dir/frame_000001.png ... as grayscale PNGs.

    Returns (fps, frame_count). Skips the work when frames_dir already holds a
    complete extraction of this exact file (same path, size and mtime). The
    marker is only written after the last frame, so an interrupted split is
    redone rather than trusted.
    """
    os.makedirs(frames_dir, exist_ok=True)
    marker = os.path.join(frames_dir, EXTRACT_MARKER)
    signature = _source_signature(video_path)

    if os.path.isfile(marker):
        try:
            with open(marker, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("signature") == signature:
                log(f"Canny: reusing {saved['count']} already-split frames in {frames_dir}")
                return float(saved.get("fps", 0.0)), int(saved["count"])
        except (OSError, ValueError, KeyError):
            pass

    # Stale or partial extraction: clear only our own frame files so a shorter
    # re-split can't leave extra frames from an older, longer one behind.
    for name in os.listdir(frames_dir):
        if FRAME_PATTERN.match(name) or name == EXTRACT_MARKER:
            os.remove(os.path.join(frames_dir, name))

    count = 0
    with av.open(video_path) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {video_path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate else 0.0
        total = int(stream.frames or 0)

        log(f"Canny: splitting {os.path.basename(video_path)} -> {frames_dir}"
            + (f" ({total} frames)" if total else ""))
        pbar = comfy.utils.ProgressBar(total) if total else None

        for frame in container.decode(stream):
            comfy_mm.throw_exception_if_processing_interrupted()
            count += 1
            _write_png(os.path.join(frames_dir, f"frame_{count:06d}.png"),
                       frame.to_ndarray(format="gray"))
            if pbar is not None:
                pbar.update_absolute(min(count, total), total)

    with open(marker, "w", encoding="utf-8") as f:
        json.dump({"signature": signature, "fps": fps, "count": count}, f, indent=2)

    log(f"Canny: split {count} frames @ {fps:.3f} fps.", "finish")
    return fps, count


# --------------------------------------------------------------------------- #
# GPU work
# --------------------------------------------------------------------------- #

def _canny_on_device(batch: torch.Tensor, low: float, high: float, device) -> torch.Tensor:
    """batch: (B, C, H, W) on CPU, C = 1 or 3, values 0..1.
    Returns (B, 1, H, W) edges on CPU. Nothing from this call stays on device."""
    with torch.inference_mode():
        x = batch.to(device=device, dtype=torch.float32)
        magnitude, edges = canny(x, low, high)
        out = edges.to("cpu")
    del x, magnitude, edges
    return out


def _edges_to_uint8(edges: torch.Tensor) -> np.ndarray:
    """(B, 1, H, W) float -> (B, H, W) uint8."""
    return (edges[:, 0].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()


def _uint8_to_image(arrays: list) -> torch.Tensor:
    """List of (H, W) uint8 -> ComfyUI IMAGE (B, H, W, 3) on the intermediate device."""
    t = torch.from_numpy(np.stack(arrays)).to(torch.float32).div_(255.0)
    t = t.unsqueeze(-1).repeat(1, 1, 1, 3)
    return t.to(device=comfy_mm.intermediate_device(), dtype=comfy_mm.intermediate_dtype())


class FG_CannyDetect:
    def __init__(self):
        self.NODE_NAME = "Canny Edges (Sequential)"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_path": ("STRING", {
                    "default": "",
                    "tooltip": "A video file, a folder of already-split frames, or a single image. "
                               "Relative paths are looked up in ComfyUI's input folder, then output. "
                               "Leave empty to use the image input instead.",
                }),
                "output_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Where edges/ (and frames/ for a video) are written. Relative = inside "
                               "ComfyUI's output folder. Empty = output/fg_canny/<source name>. "
                               "Supports %HMSf%. With the image input, nothing is written unless this is set.",
                }),
                "low_threshold": ("FLOAT", {
                    "default": 0.20, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "Hysteresis low threshold.",
                }),
                "high_threshold": ("FLOAT", {
                    "default": 0.60, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "Hysteresis high threshold. Swapped with low if set lower than it.",
                }),
                "batch_size": ("INT", {
                    "default": 5, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Frames on the device at once. Frames of different sizes are never batched together.",
                }),
                "skip_existing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip frames whose edge PNG already exists, so an interrupted run resumes. "
                               "Turn off after changing thresholds to redo everything.",
                }),
                "return_images": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also return every edge frame as an IMAGE batch. Off = only the last frame "
                               "is returned as a preview, keeping RAM flat for long videos. "
                               "Always on for the image input.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Used only when source_path is empty. Processed batch_size frames at a time.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("Edges", "Edges Folder", "Frame Count", "FPS")
    FUNCTION = "detect_edges"
    CATEGORY = "Farrenzo's Garbage/Image"
    DESCRIPTION = ("Canny edge detection for videos and frame folders that only ever holds "
                   "batch_size frames on the GPU. Videos are split to a frames folder first; "
                   "edges are written to disk as they finish.")

    @classmethod
    def IS_CHANGED(cls, source_path="", **kwargs):
        # The path string alone doesn't change when the frames on disk do.
        src = _resolve_source(source_path)
        if not src or not os.path.exists(src):
            return ""
        if os.path.isdir(src):
            latest, count = 0.0, 0
            for entry in os.scandir(src):
                if entry.is_file() and entry.name.lower().endswith(IMAGE_EXTS):
                    count += 1
                    latest = max(latest, entry.stat().st_mtime)
            return f"{count}:{latest}"
        st = os.stat(src)
        return f"{st.st_size}:{st.st_mtime}"

    # ------------------------------------------------------------------ #

    def detect_edges(self, source_path, output_folder, low_threshold, high_threshold,
                     batch_size, skip_existing, return_images, image=None):
        low, high = float(low_threshold), float(high_threshold)
        if low > high:
            log(f"Canny: low ({low}) > high ({high}); swapping.", "warning")
            low, high = high, low
        batch_size = max(1, int(batch_size))
        device = comfy_mm.get_torch_device()

        src = _resolve_source(source_path)
        if src is None:
            if image is None:
                raise ValueError(f"{self.NODE_NAME}: give it a source_path or connect an image.")
            return self._run_tensor(image, low, high, batch_size, device, output_folder)

        if not os.path.exists(src):
            raise FileNotFoundError(f"{self.NODE_NAME}: source not found -> {src}")

        fps = 0.0
        if os.path.isdir(src):
            out_dir = _resolve_output_folder(output_folder, os.path.basename(src.rstrip("\\/")))
            frames = _list_frames(src)
        else:
            stem, ext = os.path.splitext(os.path.basename(src))
            out_dir = _resolve_output_folder(output_folder, stem)
            if ext.lower() in IMAGE_EXTS:
                frames = [src]
            elif ext.lower() in VIDEO_EXTS:
                frames_dir = os.path.join(out_dir, "frames")
                fps, _ = _extract_video(src, frames_dir)
                frames = _list_frames(frames_dir)
            else:
                raise ValueError(f"{self.NODE_NAME}: unsupported file type '{ext}' -> {src}")

        if not frames:
            raise ValueError(f"{self.NODE_NAME}: no image frames found in {src}")

        edges_dir = os.path.join(out_dir, "edges")
        os.makedirs(edges_dir, exist_ok=True)

        results = self._run_frames(frames, edges_dir, low, high, batch_size, device,
                                   skip_existing, return_images)
        return (results, edges_dir, len(frames), fps)

    # ------------------------------------------------------------------ #

    def _run_frames(self, frames, edges_dir, low, high, batch_size, device,
                    skip_existing, return_images):
        total = len(frames)
        pbar = comfy.utils.ProgressBar(total)
        kept = [None] * total if return_images else None
        last = None
        processed = skipped = done = 0

        pending_idx, pending_out, pending_arr = [], [], []

        def flush():
            nonlocal processed, done, last
            if not pending_arr:
                return
            batch = torch.from_numpy(np.stack(pending_arr)).unsqueeze(1).to(torch.float32).div_(255.0)
            edges = _edges_to_uint8(_canny_on_device(batch, low, high, device))
            for idx, out_path, edge in zip(pending_idx, pending_out, edges):
                _write_png(out_path, edge)
                if kept is not None:
                    kept[idx] = edge
                last = edge
            processed += len(pending_arr)
            done += len(pending_arr)
            pbar.update_absolute(done, total)
            pending_idx.clear(); pending_out.clear(); pending_arr.clear()

        log(f"Canny: {total} frames -> {edges_dir} (batch {batch_size}, low {low}, high {high})")

        for idx, frame_path in enumerate(frames):
            comfy_mm.throw_exception_if_processing_interrupted()
            out_path = os.path.join(edges_dir, os.path.splitext(os.path.basename(frame_path))[0] + ".png")

            if skip_existing and os.path.isfile(out_path):
                # Only read it back if someone needs the pixels.
                if kept is not None or idx == total - 1:
                    edge = _read_gray(out_path)
                    if kept is not None:
                        kept[idx] = edge
                    last = edge
                skipped += 1
                done += 1
                pbar.update_absolute(done, total)
                continue

            arr = _read_gray(frame_path)
            if pending_arr and arr.shape != pending_arr[0].shape:
                flush()
            pending_idx.append(idx); pending_out.append(out_path); pending_arr.append(arr)
            if len(pending_arr) >= batch_size:
                flush()

        flush()
        comfy_mm.soft_empty_cache()

        log(f"Canny: {processed} processed, {skipped} skipped (already done).", "finish")

        if kept is not None:
            shapes = {k.shape for k in kept}
            if len(shapes) == 1:
                return _uint8_to_image(kept)
            log("Canny: frames differ in size, can't batch them into one IMAGE; "
                "returning the last frame only. Everything is still on disk.", "warning")
        return _uint8_to_image([last])

    # ------------------------------------------------------------------ #

    def _run_tensor(self, image, low, high, batch_size, device, output_folder):
        """Drop-in path for an IMAGE input: same result as core Canny, but only
        batch_size frames ever go to the device."""
        total = image.shape[0]
        pbar = comfy.utils.ProgressBar(total)

        edges_dir = ""
        if _clean_path_text(output_folder):
            edges_dir = os.path.join(_resolve_output_folder(output_folder, "image_batch"), "edges")
            os.makedirs(edges_dir, exist_ok=True)

        out = torch.empty(
            (total, image.shape[1], image.shape[2], 3),
            dtype=comfy_mm.intermediate_dtype(),
            device=comfy_mm.intermediate_device(),
        )

        for start in range(0, total, batch_size):
            comfy_mm.throw_exception_if_processing_interrupted()
            chunk = image[start:start + batch_size, :, :, :3].movedim(-1, 1)
            edges = _canny_on_device(chunk, low, high, device)  # (b, 1, H, W) CPU

            out[start:start + edges.shape[0]] = (
                edges.movedim(1, -1).repeat(1, 1, 1, 3)
                .to(device=out.device, dtype=out.dtype)
            )
            if edges_dir:
                for i, edge in enumerate(_edges_to_uint8(edges)):
                    _write_png(os.path.join(edges_dir, f"frame_{start + i + 1:06d}.png"), edge)

            pbar.update_absolute(min(start + batch_size, total), total)

        comfy_mm.soft_empty_cache()
        return (out, edges_dir, total, 0.0)
