"""
FG_LatentTransfer — move a latent from one VAE's space into another's.

Two independently-trained VAEs share no basis. For example: Anima's VAE is a 
16-channel video VAE, while SDXL's VAE is a 4-channel image. Pixels are the
only vocabulary both speak, so decode-then-encode is not a workaround, it
is the conversion. This node simply collapses it into one step.

  ANIMA LATENT ─┐
  anima vae  ───┼─> [FG_LatentTransfer] ─> SDXL LATENT ─> Illustrious KSampler
  sdxl vae   ───┘            └──────────> IMAGE ─> (save / linework composite)

┌──────────────────────────────────────────┐
│      Latent Transfer (VAE to VAE)        │
├──────────────────────────────────────────┤
│ ○ Samples                    Samples  ○  │
│ ○ Source VAE                   Image  ○  │
│ ○ Target VAE (Optional)                  │
│                                          │
│ <→Input _tile_size_>         -1, 0-4096  │
│ <→Input _overlap_>           -1, 0-4096  │
│ <→Input _temporal_size_>     -1, 0-4096  │
│ <→Input _temporal_overlap_>  -1, 0-4096  │
│                                          │
└──────────────────────────────────────────┘

TILING NOTES (from comfy/sd.py)
  * decode_tiled takes tile sizes in LATENT units; encode_tiled takes PIXELS.
    Stock VAEDecodeTiled divides by spacial_compression_decode(), stock
    VAEEncodeTiled does not. This node takes pixels everywhere and converts.
  * For 2D VAEs, decode_tiled_ / encode_tiled_ run tiled_scale THREE times
    (tile//2 x tile*2, tile*2 x tile//2, tile x tile) and average, to hide
    seams. Tiling an image VAE therefore costs roughly 3x regardless of tile
    size. decode_tiled_3d runs once, so video VAEs pay no such penalty.
  * So AUTO's first job is deciding whether to tile at all, not picking a size.
    Skipping tiling is worth far more than any tile-size tuning. Comfy's own
    decode() already grows tiles adaptively if it OOMs, so that half is covered.

  tile_size / overlap / temporal_size / temporal_overlap:
     -1  = auto  (default)
      0  = force off
     >0  = explicit, in pixels (frames for the temporal pair)
"""

import comfy.model_management as mm

AUTO = -1
# Fraction of free VRAM an untiled pass may claim before we tile instead.
_HEADROOM = 0.75
_TILE_LADDER = (2048, 1536, 1024, 768, 512, 384, 256, 192, 128)


def _bounded_shape(shape, tile_x, tile_y, tile_t):
    """Shape of a single tile, for feeding comfy's memory estimators."""
    s = list(shape)
    if len(s) == 5:      # B C T H W
        if tile_t:
            s[2] = min(s[2], tile_t)
        if tile_y:
            s[3] = min(s[3], tile_y)
        if tile_x:
            s[4] = min(s[4], tile_x)
    elif len(s) == 4:    # B C H W
        if tile_y:
            s[2] = min(s[2], tile_y)
        if tile_x:
            s[3] = min(s[3], tile_x)
    return tuple(s)


def _auto_tile(vae, shape, estimator, unit_divisor, tile_t=None):
    """Return a tile size in PIXELS, or 0 for 'do not tile'.

    shape is in the estimator's own units (latent for decode, pixel for
    encode). unit_divisor converts pixels into those units.
    """
    try:
        free = mm.get_free_memory(vae.device)
        full = estimator(shape, vae.vae_dtype)
    except Exception:
        return 0

    budget = free * _HEADROOM
    if full <= budget:
        return 0  # fits whole — always cheaper than any tiling

    longest = max(shape[-2], shape[-1]) * unit_divisor

    for px in _TILE_LADDER:
        if px > longest:
            continue
        units = max(1, px // unit_divisor)
        try:
            cost = estimator(_bounded_shape(shape, units, units, tile_t), vae.vae_dtype)
        except Exception:
            return 512
        if cost <= budget:
            return px

    return _TILE_LADDER[-1]


def _resolve(value, auto_value):
    return auto_value if value == AUTO else value


class FG_LatentTransfer:
    """Decode with one VAE, re-encode with another. Also returns the pixels."""

    @classmethod
    def INPUT_TYPES(cls):
        def auto_int(step, tip):
            return ("INT", {"default": AUTO, "min": AUTO, "max": 4096,
                            "step": step, "tooltip": tip})

        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "Latent in the source VAE's space."}),
                "source_vae": ("VAE", {"tooltip": "The VAE that produced this latent."}),
            },
            "optional": {
                "target_vae": ("VAE", {
                    "tooltip": "VAE of the model you're handing off to. Leave "
                               "unconnected to decode only; samples pass through."}),
                "tile_size": auto_int(64,
                    "Pixels. -1 auto (measures free VRAM and skips tiling entirely "
                    "when the pass fits), 0 forces off, >0 explicit."),
                "overlap": auto_int(32,
                    "Pixels. -1 auto (tile/8, the same ratio as stock's 512/64), "
                    "0 forces off. Clamped to tile/4."),
                "temporal_size": auto_int(4,
                    "Frames per chunk, video VAEs only. -1 auto (64, or off when the "
                    "latent is a single frame)."),
                "temporal_overlap": auto_int(4,
                    "Frames of temporal overlap, video VAEs only. -1 auto (8)."),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("samples", "image")
    FUNCTION = "transfer"
    CATEGORY = "Farrenzo's Garbage/Latent"
    DESCRIPTION = ("Decode a latent with one VAE and re-encode it with another, "
                   "returning the intermediate image as well.")

    # -- decode ---------------------------------------------------------

    def _decode(self, vae, latent, tile_size, overlap, temporal_size, temporal_overlap):
        compression = vae.spacial_compression_decode() or 8
        frames = latent.shape[2] if latent.ndim == 5 else 1

        t_size = _resolve(temporal_size, 0 if frames <= 1 else 64)
        t_overlap = _resolve(temporal_overlap, 8)

        size = _resolve(
            tile_size,
            _auto_tile(vae, latent.shape, vae.memory_used_decode, compression,
                       tile_t=(t_size or None)),
        )

        if size <= 0:
            return vae.decode(latent), 0

        ov = min(_resolve(overlap, max(32, size // 8)), size // 4)

        # decode_tiled wants latent units; the widget is in pixels.
        tile_lat = max(1, size // compression)
        ov_lat = max(0, ov // compression)

        t_comp = vae.temporal_compression_decode()
        if t_comp is not None and t_size:
            tt = max(2, t_size // t_comp)
            to = max(1, min(tt // 2, t_overlap // t_comp))
        else:
            tt = to = None

        return vae.decode_tiled(latent, tile_x=tile_lat, tile_y=tile_lat,
                                overlap=ov_lat, tile_t=tt, overlap_t=to), size

    # -- encode ---------------------------------------------------------

    def _encode(self, vae, pixels, tile_size, overlap, temporal_size, temporal_overlap):
        shape = (pixels.shape[0], 3, pixels.shape[1], pixels.shape[2])
        size = _resolve(tile_size, _auto_tile(vae, shape, vae.memory_used_encode, 1))

        if size <= 0:
            return vae.encode(pixels), 0

        ov = min(_resolve(overlap, max(32, size // 8)), size // 4)
        t_size = _resolve(temporal_size, 64)
        t_overlap = _resolve(temporal_overlap, 8)

        # encode_tiled already works in pixels — no conversion.
        return vae.encode_tiled(pixels, tile_x=size, tile_y=size, overlap=ov,
                                tile_t=t_size, overlap_t=t_overlap), size

    # -- node -----------------------------------------------------------

    def transfer(self, samples, source_vae, target_vae=None, tile_size=AUTO,
                 overlap=AUTO, temporal_size=AUTO, temporal_overlap=AUTO):
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]

        image, dec_tile = self._decode(source_vae, latent, tile_size, overlap,
                                       temporal_size, temporal_overlap)
        if image.ndim == 5:  # combine batches from video VAEs
            image = image.reshape(-1, *image.shape[-3:])

        if target_vae is None:
            print(f"[FG_LatentTransfer] decode only, tile={dec_tile or 'off'}, "
                  f"{tuple(latent.shape)} -> {tuple(image.shape)}")
            return (samples, image)

        mm.soft_empty_cache()

        out, enc_tile = self._encode(
            target_vae, image[..., :3], tile_size, overlap,
            temporal_size, temporal_overlap
        )

        print(f"[FG_LatentTransfer] {tuple(latent.shape)} "
              f"-decode(tile={dec_tile or 'off'})-> {tuple(image.shape)} "
              f"-encode(tile={enc_tile or 'off'})-> {tuple(out.shape)}")

        return ({"samples": out}, image)

