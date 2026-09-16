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
  * The decision logic itself now lives in _fg_helperfunctions (encode_auto /
    decode_auto) so the encode side of the pack shares it. This node just
    passes its four widgets through.
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
from ._fg_helperfunctions import AUTO, encode_auto, decode_auto


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

    # -- node -----------------------------------------------------------

    def transfer(self, samples, source_vae, target_vae=None, tile_size=AUTO,
                 overlap=AUTO, temporal_size=AUTO, temporal_overlap=AUTO):
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]

        image, dec_tile = decode_auto(source_vae, latent, tile_size, overlap,
                                      temporal_size, temporal_overlap)
        if image.ndim == 5:  # combine batches from video VAEs
            image = image.reshape(-1, *image.shape[-3:])

        if target_vae is None:
            print(f"[FG_LatentTransfer] decode only, tile={dec_tile or 'off'}, "
                  f"{tuple(latent.shape)} -> {tuple(image.shape)}")
            return (samples, image)

        mm.soft_empty_cache()

        # encode_auto trims to 3 channels itself.
        out, enc_tile = encode_auto(
            target_vae, image, tile_size, overlap,
            temporal_size, temporal_overlap
        )

        print(f"[FG_LatentTransfer] {tuple(latent.shape)} "
              f"-decode(tile={dec_tile or 'off'})-> {tuple(image.shape)} "
              f"-encode(tile={enc_tile or 'off'})-> {tuple(out.shape)}")

        return ({"samples": out}, image)
