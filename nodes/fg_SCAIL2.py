"""
SCAIL-2 Easy Config — a single control node for the SCAIL-2 GGUF auto workflow.
Plus SCAIL2AutoVideo — the segment LOOP node that runs exactly the number of
segments needed for the requested seconds (1 for <=5 s, 3 for 10 s, 7 for 30 s)
instead of always iterating a fixed 7-segment chain.
"""

import json
import os
import random
import struct
import subprocess
import zlib

import torch
from comfy_extras.nodes_scail import WanSCAILToVideo
from comfy_extras.nodes_custom_sampler import SamplerCustom
from nodes import VAEDecode

try:
    import folder_paths
    _INPUT_DIR = os.path.join(folder_paths.base_path, "input")
except Exception:
    _INPUT_DIR = None


def _probe_video_dims(video):
    """Return (width, height) of the driving video by probing with ffprobe.
    None if it cannot be found/probed."""
    candidates = []
    if video:
        candidates.append(video)
        if not os.path.isabs(video):
            if _INPUT_DIR:
                candidates.append(os.path.join(_INPUT_DIR, video))
            # relative to this node file: custom_nodes/scail2-easy-config -> ComfyUI/input
            candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "input", video))
    for p in candidates:
        if not os.path.isfile(p):
            continue
        try:
            out = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=s=x:p=0", p,
                ],
                capture_output=True, text=True, timeout=20,
            )
            dim = out.stdout.strip()
            if "x" in dim:
                w, h = dim.split("x")[:2]
                return int(w), int(h)
        except Exception:
            pass
    return None


def _round32(v):
    return max(256, min(1280, int(round(v / 32.0)) * 32))


class SCAIL2EasyConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seconds": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 0.0,
                        "max": 120.0,
                        "step": 0.5,
                        "tooltip": "How many seconds of the driving video to animate. 0 = whole video.",
                    },
                ),
                "fps": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 60,
                        "tooltip": "Output frame rate. Wan native = 16. 20/22/24 work too (model is trained with mixed fps) but produce more frames per second -> more VRAM/time.",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 832,
                        "min": 0,
                        "max": 2048,
                        "step": 32,
                        "tooltip": "Generation width (rounded to a multiple of 32). 0 = auto: same width as the driving video (or proportional to the height you set).",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 480,
                        "min": 0,
                        "max": 2048,
                        "step": 32,
                        "tooltip": "Generation height (rounded to a multiple of 32). 0 = auto: same height as the driving video (or proportional to the width you set).",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 60,
                        "tooltip": "Sampling steps. 6 = the LightX2V distill setting.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "CFG scale. 1.0 is the distill setting.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Seed for ALL segments when randomize_seed is off.",
                    },
                ),
                "replace_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "False = Animation Mode (default): renders the FULL reference image - ALL subjects in it - performing the video's motion. True = Replacement Mode: swaps only the tracked person in the video.",
                    },
                ),
                "object_indices": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Which tracked person(s) to use, 0-based, comma-separated. Empty = all.",
                    },
                ),
                "randomize_seed": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "ON: every Queue run picks ONE random seed, shared by ALL segments. OFF: all segments use the seed value above.",
                    },
                ),
                "video": (
                    "STRING",
                    {
                        "default": "your_driving_video.mp4",
                        "tooltip": "Driving video filename (in ComfyUI/input). Only used when width or height is 0 to auto-pick the resolution.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "FLOAT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("frames", "fps", "width", "height", "steps", "cfg", "seed", "replace_mode", "object_indices")
    FUNCTION = "go"
    CATEGORY = "SCAIL-2"
    DESCRIPTION = (
        "One node that controls every SCAIL-2 parameter. Outputs frames (= seconds x fps), "
        "fps, width, height, steps, cfg, seed, replace_mode and object_indices. "
        "Width/height = 0 auto-uses the driving video resolution."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("randomize_seed"):
            return float("NaN")
        return None

    def go(self, seconds, fps, width, height, steps, cfg, seed, replace_mode, object_indices, randomize_seed, video):
        fps = int(fps)
        frames = int(round(seconds * fps)) if seconds > 0 else 0
        # Guard: object_indices is a comma-separated INDEX LIST ("" = all, "0,2"), not a
        # boolean. A stray "True"/"False" makes SCAIL2ColoredMask select ZERO objects ->
        # empty masks -> the model receives no conditioning and hallucinates.
        if str(object_indices).strip().lower() in ("true", "false"):
            print("[SCAIL2EasyConfig] WARNING: object_indices=%r looks like a boolean, not an index list. "
                  "Setting it to '' (all people). Use \"0\" for the leftmost person only." % object_indices)
            object_indices = ""
        if randomize_seed:
            seed = int(random.randint(0, 2**31 - 1))
        else:
            seed = int(seed)
        w, h = int(width), int(height)
        dims = _probe_video_dims(str(video)) if (w == 0 or h == 0) else None
        if dims is not None and w == 0 and h == 0:
            w, h = dims
        elif dims is not None and w == 0:
            w = int(round(h * dims[0] / float(dims[1])))
        elif dims is not None and h == 0:
            h = int(round(w * dims[1] / float(dims[0])))
        w = _round32(w)
        h = _round32(h)
        print("[SCAIL2EasyConfig] seconds=%s fps=%d -> frames=%d | res=%dx%d | steps=%d cfg=%s | replace_mode=%s | randomize_seed=%s | seed=%d"
              % (seconds, fps, frames, w, h, int(steps), float(cfg), bool(replace_mode), bool(randomize_seed), seed))
        return (frames, fps, w, h, int(steps), float(cfg), seed, bool(replace_mode), object_indices)


class SCAIL2AutoVideo:
    """Generate the video in a Python loop, running ONLY the segments needed
    for `frames` (0 = whole driving video). 81-frame chunks, 76-frame step,
    5-frame anchor - same math as the official Extend blueprint, but without
    wasting time on stub segments and without reloading the model per segment."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "reference_image": ("IMAGE",),
                "pose_video": ("IMAGE",),
                "pose_video_mask": ("IMAGE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "frames": ("INT", {"default": 160, "min": 0, "tooltip": "Total frames to generate (seconds x fps). 0 = whole driving video."}),
                "video_frames": ("INT", {"default": 480, "min": 1, "tooltip": "Loaded frame count of the driving video (VHS_VideoInfo). Used when frames = 0."}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "replacement_mode": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 832, "min": 0, "step": 32}),
                "height": ("INT", {"default": 480, "min": 0, "step": 32}),
                "length": ("INT", {"default": 81, "min": 5, "max": 512, "step": 4, "tooltip": "Max frames per segment (81 = 5 s @ 16 fps)."}),
                "previous_frame_count": ("INT", {"default": 5, "min": 1, "max": 32, "step": 4, "tooltip": "Tail frames of the previous segment used as anchor (SCAIL-2 trained at 5)."}),
            },
            "optional": {
                "reference_image_mask": ("IMAGE",),
                "anchor_frames": ("IMAGE", {"tooltip": "Tail frames of a PREVIOUS run's output (>= previous_frame_count) to anchor this run's first segment - set VHS_LoadVideo 'skip_first_frames' to the logged 'first unused source frame' of the previous run to continue the video. Leave unconnected for a fresh run."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "go"
    CATEGORY = "SCAIL-2"
    DESCRIPTION = (
        "Runs the SCAIL-2 extend chain in a loop, generating exactly the number of "
        "segments needed for the requested frame count. 10 s -> 3 segments, 30 s -> 7, "
        "5 s -> 1. Output is trimmed/padded to exactly `frames`."
    )

    @staticmethod
    def _plan(frames, pose_len, length=81, pc=5, anchored=False):
        """Return a list of (offset_out, chunk_length) pairs - the loop plan.

        offset_out is the video_frame_offset handed to WanSCAILToVideo. With
        previous_frames connected the effective source window is
        [offset_out - pc, offset_out - pc + length - 1]; without, [offset_out, offset_out + length - 1].
        anchored=True means the FIRST chunk re-renders the seam on top of the
        previous run's anchor frames, so it only produces length - pc new frames.
        """
        frames = max(1, int(frames))
        length = max(1, int(length))
        pc = max(1, int(pc))

        def l4k1(L):
            L = max(1, int(L))
            return ((L - 1) // 4) * 4 + 1

        plan = []
        offset_out = 0
        produced = 0
        chunk_i = 0
        while chunk_i < 64:
            if chunk_i == 0:
                target = frames + (pc if anchored else 0)
                L = l4k1(min(length, target, pose_len))
                if anchored and L - pc < 1:
                    L = l4k1(min(pc + 4, pose_len))
                if L < 1:
                    break
                plan.append((offset_out, L))
                produced = (L - pc) if anchored else L
                offset_out = L
            else:
                pose_start = offset_out - pc
                if pose_start < 0 or pose_start >= pose_len:
                    break
                remaining_src = pose_len - pose_start
                # cap so this chunk doesn't overshoot the target: new frames = L - pc
                L = l4k1(min(length, remaining_src, frames - produced + pc))
                if L - pc < 1:
                    break  # anchor eats the whole chunk -> nothing new, stop
                plan.append((offset_out, L))
                produced = produced + L - pc
                offset_out = pose_start + L
            chunk_i += 1
            if produced >= frames:
                break
        return plan

    def go(self, model, vae, positive, negative, clip_vision_output, reference_image,
           pose_video, pose_video_mask, sampler, sigmas, frames, video_frames, seed, cfg,
           replacement_mode, width, height, length=81, previous_frame_count=5,
           reference_image_mask=None, anchor_frames=None):
        if frames <= 0:
            frames = int(video_frames) if video_frames and video_frames > 0 else 81
        frames = max(1, int(frames))
        length = max(1, int(length))
        pc = max(1, int(previous_frame_count))
        pose_len = pose_video.shape[0] if pose_video is not None else frames
        anchored = anchor_frames is not None and anchor_frames.shape[0] > 0
        if anchored and anchor_frames.shape[0] > pc:
            anchor_frames = anchor_frames[-pc:]
        plan = self._plan(frames, pose_len, length, pc, anchored)

        print("[SCAIL2AutoVideo] seed=%d | frames=%d | pose_video_frames=%d | length=%d pc=%d | chunks=%d | anchored=%s"
              % (int(seed), frames, pose_len, length, pc, len(plan), anchored))

        chunks = []
        last_first = 0
        last_last = 0
        for i, (offset_in, L) in enumerate(plan):
            prev = chunks[-1] if chunks else None
            off_call = offset_in
            if i == 0 and anchored:
                off_call = offset_in + pc
                prev = anchor_frames
            res = WanSCAILToVideo.execute(
                positive, negative, vae, int(width), int(height), L, 1, 1.0, 0.0, 1.0,
                off_call, pc, bool(replacement_mode),
                reference_image=reference_image,
                clip_vision_output=clip_vision_output,
                pose_video=pose_video,
                pose_video_mask=pose_video_mask,
                reference_image_mask=reference_image_mask,
                previous_frames=prev,
            )
            first = (off_call - pc) if prev is not None else off_call
            last = first + L - 1
            last_first, last_last = first, last
            new_frames = L if prev is None else L - pc  # anchored chunks drop their anchor frames
            print("[SCAIL2AutoVideo]   chunk %d: %d new frames (L=%d%s) | drives source frames %d..%d | video_frame_offset=%d"
                  % (i, new_frames, L, " incl. %d anchor" % pc if prev is not None else "", first, last, off_call))
            latent = res[2]
            # CRITICAL: WanSCAILToVideo.execute attaches ALL the SCAIL-2 conditioning
            # (reference_latents, clip_vision, pose_video_latent, driving_mask_28ch,
            # ref_mask_28ch) to the conditioning it RETURNS (res[0]/res[1]). Sampling
            # with the original text-only conditioning makes the model ignore the
            # reference image, pose video and masks entirely -> garbage output.
            cond_pos, cond_neg = res[0], res[1]
            if hasattr(cond_pos, "result"):
                cond_pos, cond_neg = cond_pos.result[0], cond_neg.result[0]
            out = SamplerCustom.execute(model, True, int(seed), float(cfg),
                                        cond_pos, cond_neg, sampler, sigmas, latent)
            # ComfyUI 0.33+: execute returns io.NodeOutput whose first output is
            # already the LATENT dict {"samples": tensor}; older builds return a
            # plain tuple. Handle both, and do NOT wrap it again.
            if hasattr(out, "result"):
                out = out.result
            samples = out[0]
            imgs = VAEDecode().decode(vae, samples)[0]
            if prev is not None and imgs.shape[0] > 0:
                # The first `pc` frames of an anchored chunk are the anchor itself
                # (copies of the previous chunk's tail). Drop them so the final
                # video has no replayed frames at the seam. (Matches _plan's
                # accounting: anchored chunks produce L - pc new frames.)
                imgs = imgs[min(pc, imgs.shape[0]):]
            chunks.append(imgs)

        if not chunks:
            raise RuntimeError("SCAIL2AutoVideo: no segments planned (frames=%d pose=%d)" % (frames, pose_len))

        all_imgs = torch.cat(chunks, dim=0)
        if all_imgs.shape[0] > frames:
            all_imgs = all_imgs[:frames]
        elif all_imgs.shape[0] < frames:
            pad = all_imgs[-1:].expand(frames - all_imgs.shape[0], -1, -1, -1)
            all_imgs = torch.cat([all_imgs, pad], dim=0)

        print("[SCAIL2AutoVideo] done: %d frames out (%d requested) | source video frames used: %d..%d | first unused source frame: %d"
              % (all_imgs.shape[0], frames, 0 if plan else 0, last_last, last_last + 1))
        print("[SCAIL2AutoVideo] to continue in a NEW run: set VHS_LoadVideo 'skip_first_frames'=%d (keep seconds/frame_load_cap) and feed this run's last %d output frames into the loop node's 'anchor_frames' input"
              % (last_last + 1, pc))
        return (all_imgs,)


class SCAIL2RunInfo:
    """Read a ComfyUI output file (video or image) and report everything
    recoverable from its embedded metadata: seed, frames, fps, resolution,
    steps, cfg, prompts, mode and source files. Works on SCAIL2_FULL_*.mp4
    (VHS writes the prompt + workflow JSON into the mp4 'comment' tag) and on
    ComfyUI-saved PNGs (tEXt/iTXt 'prompt'/'workflow' chunks)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": ("STRING", {"default": "", "tooltip": "Filename or path of a ComfyUI output (mp4/webm/png/jpg/webp). Bare names are looked up in output/ then input/. e.g. SCAIL2_FULL_00001.mp4"}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "INT", "INT", "INT", "FLOAT",
                    "STRING", "STRING", "BOOLEAN", "STRING", "STRING", "STRING",
                    "STRING", "STRING")
    RETURN_NAMES = ("seed", "frames", "fps", "width", "height", "steps", "cfg",
                    "positive", "negative", "replace_mode", "object_indices",
                    "driving_video", "reference_image", "file", "message")
    FUNCTION = "go"
    CATEGORY = "SCAIL-2"
    DESCRIPTION = (
        "Read a ComfyUI output (mp4/webm/png/jpg/webp) and extract all recoverable run "
        "info from its embedded metadata: seed, frames, fps, resolution, steps, cfg, "
        "positive/negative prompts, mode, driving video and reference image. "
        "Note: when randomize_seed was ON the ACTUAL seed is not stored in the file - "
        "it is printed to the ComfyUI console instead."
    )

    # ---------------------------------------------------------------- file io
    @staticmethod
    def _resolve_path(name):
        if not name:
            return None
        if os.path.isabs(name) and os.path.isfile(name):
            return name
        if os.path.isfile(name):
            return os.path.abspath(name)
        base = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base, "..", "..", "output", name),
            os.path.join(base, "..", "..", "input", name),
        ]
        try:
            import folder_paths
            candidates.insert(0, os.path.join(folder_paths.get_output_directory(), name))
            candidates.insert(1, os.path.join(folder_paths.get_input_directory(), name))
        except Exception:
            pass
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    @staticmethod
    def _png_text_chunks(path):
        """Return {keyword: text} from tEXt/iTXt chunks of a PNG file."""
        out = {}
        try:
            with open(path, "rb") as f:
                if f.read(8) != b"\x89PNG\r\n\x1a\n":
                    return out
                while True:
                    hdr = f.read(8)
                    if len(hdr) < 8:
                        break
                    (length,) = struct.unpack(">I", hdr[:4])
                    ctype = hdr[4:8]
                    data = f.read(length)
                    f.read(4)  # crc
                    if ctype == b"tEXt":
                        kw, _, txt = data.partition(b"\x00")
                        out[kw.decode("latin-1")] = txt.decode("utf-8", "replace")
                    elif ctype == b"iTXt":
                        kw, _, rest = data.partition(b"\x00")
                        if len(rest) < 2:
                            continue
                        comp_flag = rest[0]
                        rest2 = rest[2:].partition(b"\x00")[2]  # skip lang tag + translated keyword
                        rest2 = rest2.partition(b"\x00")[2]
                        try:
                            txt = zlib.decompress(rest2) if comp_flag else rest2
                            out[kw.decode("latin-1")] = txt.decode("utf-8", "replace")
                        except Exception:
                            pass
                    if ctype == b"IEND":
                        break
        except Exception:
            pass
        return out

    @staticmethod
    def _png_dims(path):
        """Return (width, height) from a PNG's IHDR (bytes 16-23)."""
        try:
            with open(path, "rb") as f:
                f.seek(16)
                head = f.read(8)
            if len(head) == 8:
                return struct.unpack(">II", head)
        except Exception:
            pass
        return (0, 0)

    @staticmethod
    def _ffprobe(path):
        info = {"tags": {}, "stream": {}}
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", path],
                capture_output=True, text=True, timeout=30,
            )
            if out.returncode == 0:
                data = json.loads(out.stdout or "{}")
                info["tags"] = (data.get("format") or {}).get("tags") or {}
        except Exception:
            pass
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,nb_frames,avg_frame_rate,r_frame_rate,codec_name,duration",
                 "-of", "json", path],
                capture_output=True, text=True, timeout=30,
            )
            if out.returncode == 0:
                data = json.loads(out.stdout or "{}")
                streams = data.get("streams") or []
                if streams:
                    info["stream"] = streams[0]
        except Exception:
            pass
        return info

    # ------------------------------------------------------------- parsing
    @staticmethod
    def _parse_json(raw):
        try:
            d = json.loads(raw)
            return d if isinstance(d, dict) else None
        except Exception:
            return None

    def _find_class(self, prompt, workflow, cls):
        for nid, node in prompt.items():
            if node.get("class_type") == cls:
                return str(nid), node
        if workflow:
            for node in workflow.get("nodes", []):
                if node.get("type") == cls:
                    return node.get("id"), node
        return None, None

    @staticmethod
    def _wf_widgets(node):
        w = node.get("widgets_values")
        if not isinstance(w, (list, tuple)):
            return {}
        keys = ["seconds", "fps", "width", "height", "steps", "cfg", "seed",
                "replace_mode", "object_indices", "randomize_seed", "video"]
        return {k: w[i] for i, k in enumerate(keys) if i < len(w)}

    def _config_values(self, prompt, workflow):
        nid, node = self._find_class(prompt, workflow, "SCAIL2EasyConfig")
        vals = {}
        if node is None:
            return vals, False
        if isinstance(node.get("inputs"), dict):  # API prompt style (widgets -> inputs)
            ins = node["inputs"]
            for k in ("seconds", "fps", "width", "height", "steps", "cfg",
                      "seed", "replace_mode", "object_indices", "randomize_seed", "video"):
                if k in ins:
                    vals[k] = ins[k]
        else:  # UI workflow style (widgets_values list)
            vals = self._wf_widgets(node)
        return vals, True

    def _trace_conditioning_text(self, prompt, start_id, key):
        """Follow a conditioning link chain (SamplerCustom/WanSCAILToVideo/loop)
        until a CLIPTextEncode node is reached and return its text."""
        seen = set()
        cur = str(start_id)
        while cur not in seen:
            seen.add(cur)
            node = prompt.get(cur)
            if not node:
                return ""
            cls = node.get("class_type", "")
            ins = node.get("inputs") or {}
            if cls == "CLIPTextEncode" or "TextEncode" in cls:
                for tkey in ("text", "prompt", "caption"):
                    if isinstance(ins.get(tkey), str) and ins[tkey]:
                        return ins[tkey]
                return ""
            val = ins.get(key)
            if isinstance(val, list) and len(val) >= 2:
                cur = str(val[0])
            else:
                return ""
        return ""

    def _resolve_input(self, prompt, node, key, fallback=None):
        """Return a node input, resolving one level of link if needed."""
        ins = node.get("inputs") or {}
        val = ins.get(key, fallback)
        if isinstance(val, list) and len(val) >= 2:
            nid = str(val[0])
            tgt = prompt.get(nid)
            if tgt and isinstance(tgt.get("inputs"), dict):
                if tgt.get("class_type") == "PrimitiveInt":
                    return tgt["inputs"].get("value", val)
                if tgt.get("class_type") == "SCAIL2EasyConfig" and "seed" in tgt["inputs"]:
                    return tgt["inputs"]["seed"]
                return val  # can't resolve further generically
        return val

    # ---------------------------------------------------------------- main
    def go(self, file):
        path = self._resolve_path(file)
        msg = []
        info = {"seed": -1, "frames": 0, "fps": 0.0, "width": 0, "height": 0,
                "steps": 0, "cfg": 0.0, "positive": "", "negative": "",
                "replace_mode": False, "object_indices": "", "randomize": None,
                "driving_video": "", "reference_image": ""}
        prompt = {}
        workflow = None

        if path is None:
            msg.append("file not found: %r (searched output/ and input/)" % file)
            return (info["seed"], info["frames"], info["fps"], info["width"], info["height"],
                    info["steps"], info["cfg"], info["positive"], info["negative"], info["replace_mode"],
                    info["object_indices"], "", "", file, "; ".join(msg))

        ext = os.path.splitext(path)[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".webp"):
            chunks = self._png_text_chunks(path)
            prompt = self._parse_json(chunks.get("prompt")) or {}
            workflow = self._parse_json(chunks.get("workflow"))
            msg.append("image: %s" % ext)
            if ext == ".png":
                w, h = self._png_dims(path)
                info["width"], info["height"] = w, h
            info["frames"] = 1
        elif ext in (".mp4", ".webm", ".mov", ".mkv", ".avi"):
            fp = self._ffprobe(path)
            comment = (fp.get("tags") or {}).get("comment")
            if comment:
                md = self._parse_json(comment)
                if md:
                    prompt = self._parse_json(md.get("prompt")) or {}
                    workflow = self._parse_json(md.get("workflow"))
                    if not prompt and not workflow:
                        msg.append("no embedded ComfyUI metadata (comment tag present but unparsable)")
            else:
                msg.append("no embedded metadata found - VHS 'save_metadata' must be ON")
            st = fp.get("stream") or {}
            if st.get("width"):
                info["width"] = int(st["width"])
            if st.get("height"):
                info["height"] = int(st["height"])
            if st.get("nb_frames"):
                try:
                    info["frames"] = int(float(st["nb_frames"]))
                except Exception:
                    pass
            rate = st.get("avg_frame_rate") or st.get("r_frame_rate")
            if rate and "/" in str(rate):
                try:
                    num, den = str(rate).split("/")
                    info["fps"] = float(num) / float(den) if float(den) else 0.0
                except Exception:
                    pass
            msg.append("video: %s" % st.get("codec_name", ext))
        else:
            msg.append("unsupported file type: %s" % ext)

        # ---- SCAIL-2 config node
        cfg_vals, has_cfg = self._config_values(prompt, workflow)
        if has_cfg:
            info["steps"] = int(cfg_vals.get("steps", 6) or 6)
            info["cfg"] = float(cfg_vals.get("cfg", 1.0) or 1.0)
            info["replace_mode"] = bool(cfg_vals.get("replace_mode", False))
            info["object_indices"] = str(cfg_vals.get("object_indices", "") or "")
            info["randomize"] = bool(cfg_vals.get("randomize_seed", False))
            if cfg_vals.get("video"):
                info["driving_video"] = str(cfg_vals["video"])

        # ---- loop / seed
        loop_id, loop = self._find_class(prompt, workflow, "SCAIL2AutoVideo")
        seed = None
        if loop and isinstance(loop.get("inputs"), dict):
            seed = self._resolve_input(prompt, loop, "seed")
            if isinstance(seed, list):
                seed = None
            if "frames" in loop["inputs"] and not info["frames"]:
                v = loop["inputs"]["frames"]
                if isinstance(v, int):
                    info["frames"] = v
        if seed is None and has_cfg:
            s = cfg_vals.get("seed")
            seed = s if isinstance(s, int) else None
        if seed is None and not has_cfg:
            # fall back: any SamplerCustom noise_seed / KSampler seed
            for nid, node in prompt.items():
                ins = node.get("inputs") or {}
                cls = node.get("class_type")
                if cls == "SamplerCustom" and isinstance(ins.get("noise_seed"), int):
                    seed = ins["noise_seed"]
                    break
                if cls in ("KSampler", "KSamplerAdvanced") and isinstance(ins.get("seed"), int):
                    seed = ins["seed"]
                    break
        if isinstance(seed, int):
            info["seed"] = seed
        elif has_cfg and info["randomize"]:
            info["seed"] = -1
            msg.append("randomize_seed was ON - actual seed NOT stored in the file (see ComfyUI console log)")

        # ---- prompts via conditioning trace
        trace_start = loop_id
        if trace_start is None:
            for nid, node in prompt.items():
                if node.get("class_type") in ("SamplerCustom", "KSampler", "KSamplerAdvanced"):
                    trace_start = nid
                    break
        if trace_start is not None:
            info["positive"] = self._trace_conditioning_text(prompt, trace_start, "positive")
            info["negative"] = self._trace_conditioning_text(prompt, trace_start, "negative")
        if not info["positive"] and workflow:  # fallback by node title
            for node in workflow.get("nodes", []):
                t = str(node.get("title") or "").lower()
                if node.get("type") == "CLIPTextEncode" and "positive" in t:
                    w = node.get("widgets_values")
                    info["positive"] = str(w[0]) if w else ""
                elif node.get("type") == "CLIPTextEncode" and "negative" in t:
                    w = node.get("widgets_values")
                    info["negative"] = str(w[0]) if w else ""

        # ---- driving video / reference image
        for nid, node in prompt.items():
            cls = node.get("class_type", "")
            ins = node.get("inputs") or {}
            if cls == "VHS_LoadVideo" and not info["driving_video"]:
                info["driving_video"] = str(ins.get("video", ""))
            elif cls == "LoadImage" and not info["reference_image"]:
                info["reference_image"] = str(ins.get("image", ""))

        # ---- steps/cfg fallback from scheduler/samplers
        if not info["steps"] or not info["cfg"]:
            for nid, node in prompt.items():
                ins = node.get("inputs") or {}
                cls = node.get("class_type")
                if cls == "BasicScheduler" and not info["steps"]:
                    try:
                        info["steps"] = int(ins.get("steps", 0) or 0)
                    except Exception:
                        pass
                elif cls in ("KSampler", "KSamplerAdvanced") and not info["steps"]:
                    try:
                        info["steps"] = int(ins.get("steps", 0) or 0)
                    except Exception:
                        pass
                if cls == "SamplerCustom" and not info["cfg"]:
                    try:
                        info["cfg"] = float(ins.get("cfg", 0.0) or 0.0)
                    except Exception:
                        pass
                elif cls in ("KSampler", "KSamplerAdvanced") and not info["cfg"]:
                    try:
                        info["cfg"] = float(ins.get("cfg", 0.0) or 0.0)
                    except Exception:
                        pass

        mode = "Replacement" if info["replace_mode"] else "Animation"
        seed_txt = str(info["seed"]) if info["seed"] >= 0 else "? (randomized)"
        msg.insert(0, "%s | %dx%d @%g fps %d frames | seed=%s | steps=%d cfg=%g | %s"
                    % (os.path.basename(path), info["width"], info["height"], info["fps"],
                       info["frames"], seed_txt, info["steps"], info["cfg"], mode))
        if info["driving_video"]:
            msg.append("driving=%s" % info["driving_video"])
        if info["reference_image"]:
            msg.append("ref=%s" % info["reference_image"])
        if info["positive"]:
            msg.append("positive=%s" % info["positive"][:200])
        if info["negative"]:
            msg.append("negative=%s" % info["negative"][:200])

        print("[SCAIL2RunInfo] " + " | ".join(msg))
        return (info["seed"], info["frames"], info["fps"], info["width"], info["height"],
                info["steps"], info["cfg"], info["positive"], info["negative"], info["replace_mode"],
                info["object_indices"], info["driving_video"], info["reference_image"], path, "; ".join(msg))

