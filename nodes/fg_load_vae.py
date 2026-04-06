"""
Load VAE Node for ComfyUI
Provides you with a choice of VAE to use.

┌─────────────────────────────────────┐
│       Custom VAE Loader             │
├─────────────────────────────────────┤
│ ○ VAE (Optional)             VAE ○  │
│                                     │
│ <▼DROPDOWN_1>                       │
│ ○ List of available VAE's           │
│                                     │
│ <Switch | Use selected VAE>         │
│ - DEFAULT | True                    │
│                                     │
└─────────────────────────────────────┘

"""

import os
import torch
import folder_paths
import comfy.utils
import comfy.sd
import comfy.utils


class FG_VAELoader:
    def __init__(self):
        self.NODE_NAME = "Custom VAE Loader"

    video_taes = ["taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5"]
    image_taes = ["taesd", "taesdxl", "taesd3", "taef1"]

    @staticmethod
    def vae_list(s):
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False
        f1_taesd_enc = False
        f1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
            elif v.startswith("taef1_encoder."):
                f1_taesd_dec = True
            elif v.startswith("taef1_decoder."):
                f1_taesd_enc = True
            else:
                for tae in s.video_taes:
                    if v.startswith(tae):
                        vaes.append(v)

        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append("taef1")
        vaes.append("pixel_space")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes))
        decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))

        enc = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        dec = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder))
        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        elif name == "taef1":
            sd["vae_scale"] = torch.tensor(0.3611)
            sd["vae_shift"] = torch.tensor(0.1159)
        return sd

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "load_vae_from_list": ("BOOLEAN", {"default": True, "label_on": "Select below", "label_off": "External VAE"}),
                "vae_name": (s.vae_list(s),),
            },
            "optional": {
                "external_vae": ("VAE", {"tooltip": "Pass a VAE in here from checkpoint loader or another VAE loader."})
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "Farrenzo's Garbage/Loaders"
    DESCRIPTION = "Load a VAE or pass on a pre-loaded VAE from a different source like the checkpoint."

    def load_vae(self, vae_name, load_vae_from_list, external_vae=None):
        if load_vae_from_list:
            if vae_name == "pixel_space":
                sd = {}
                sd["pixel_space_vae"] = torch.tensor(1.0)
            elif vae_name in self.image_taes:
                sd = self.load_taesd(vae_name)
            else:
                if os.path.splitext(vae_name)[0] in self.video_taes:
                    vae_path = folder_paths.get_full_path_or_raise("vae_approx", vae_name)
                else:
                    vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
                sd = comfy.utils.load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=sd)
        else:
            vae = external_vae

        vae.throw_exception_if_invalid()
        return (vae,)
