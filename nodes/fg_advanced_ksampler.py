"""
"""
import math
import torch
import latent_preview

import comfy.utils       as c_utils
import comfy.sample      as c_sample
import comfy.samplers    as c_samplers
import comfy.comfy_types as c_types
import comfy.model_management

from ._fg_helperfunctions import log


global_description = """
Integrated KSampler for SDXL + FLUX models. Supports:
 - Optimized AuraFlow shift
 - Optimized CFG normalization adjustment
"""

class FG_Advanced_KSampler:
    def __init__(self):
        self.NODE_NAME = "Advanced KSampler"
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (c_types.IO.MODEL, {"tooltip": "🤖 Model: Diffusion Model input used for Image Generation"}),
                "positive": (c_types.IO.CONDITIONING, {"tooltip": "➕ Positive prompt conditioning info."}),
                "negative": (c_types.IO.CONDITIONING, {"tooltip": "➖ Negative prompt conditioning info."}),
                "latent_image": (c_types.IO.LATENT, {"tooltip": "🏞️ Latent: For I2I tasks (where a primary image is passed), providing an input image is optional; one will be generated automatically if omitted. If you wish to utilize tools such as ControlNet, you may provide the necessary inputs manually."}),

                "compute_sigmas": (c_types.IO.BOOLEAN, {"default": False, "label_on": "Latent sigma calc.", "label_off": "Default Scheduler"}),
                "add_noise":  (c_types.IO.BOOLEAN, {"default": True, "label_on": "Will add noise.", "label_off": "Will use latent"}),
                "noise_seed": (c_types.IO.INT, {"default": 42, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "🎲 The random seed for generating noise."}),
                "steps": (c_types.IO.INT, {"default": 25, "min": 1, "max": 10000, "tooltip": "📊 Noise reduction steps."}),
                "cfg":   (c_types.IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "🎛️ Used to balance randomness and prompt adherence. Increasing this value makes the results align more closely with the prompt, but setting it too high may lead to a decline in image quality."}),
                "sampler_name": (c_samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "🌀 Sampling algorithms influence result quality, generation speed, and stylistic characteristics."}),
                "scheduler":    (c_samplers.KSampler.SCHEDULERS, {"default": "simple", "tooltip": "📈 A method for controlling the gradual removal of noise."}),
                "start_at_step": (c_types.IO.INT, {"default": 0, "min": 0, "max": 10000, "advanced": True}),
                "end_at_step":   (c_types.IO.INT, {"default": 10000, "min": 0, "max": 10000, "advanced": True}),

                "denoise":           (c_types.IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "🔄 Noise Reduction Strength: Lowering this value preserves a significant portion of the original image content, thereby enabling image-to-image generation."}),
                "auraflow_shift":    (c_types.IO.FLOAT, {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "⚡ AuraFlow Sampling Algorithm Shift — Sampling Algorithm (AuraFlow) Shift Parameter: Affects Speed and Quality (0-100)"}),
                "cfg_norm_strength": (c_types.IO.FLOAT, {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "⚖️ CFGNorm Strength: CFG Normalization Strength; dynamically adjusts CFG guidance strength. (0-100)"}),

                "return_with_leftover_noise": (c_types.IO.BOOLEAN, {"default": False, "label_on": "Will return some noise.", "label_off": "Will denoise completely"})
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("Latent")
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"
    CATEGORY = "Farrenzo's Garbage/Sampling"
    DESCRIPTION = global_description
    SEARCH_ALIASES = ["FG", "sampler", "sample", "generate", "denoise", "diffuse", "txt2img", "img2img"]

    def _aura_flow_shift(self, model, auraflow_shift):
        import comfy.model_sampling
        sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
        sampling_type = comfy.model_sampling.CONST
        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=auraflow_shift, multiplier=1.0)
        model.add_object_patch("model_sampling", model_sampling)
        return model

    def _cfg_normalizer(self, model, cfg_norm_strength):
        def _norm(args):
            cond_p = args['cond_denoised']
            pred_text_ = args["denoised"]
            norm_full_cond = torch.norm(cond_p, dim=1, keepdim=True)
            norm_pred_text = torch.norm(pred_text_, dim=1, keepdim=True)
            scale = (norm_full_cond / (norm_pred_text + 1e-8)).clamp(min=0.0, max=1.0)
            return pred_text_ * scale * cfg_norm_strength
        model.set_model_sampler_post_cfg_function(_norm)
        return model

    def _get_sigmas(self, steps, width, height):
        seq_len = (width * height / (16 * 16))
        sigmas = get_schedule(steps, round(seq_len))
        return sigmas

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        compute_sigmas,
        add_noise,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        start_at_step,
        end_at_step,
        denoise,
        auraflow_shift,
        cfg_norm_strength,
        return_with_leftover_noise,
    ):
        # Preliminary option preparations
        disable_noise = False
        if not add_noise:
            disable_noise = True

        force_full_denoise = True
        if return_with_leftover_noise:
            force_full_denoise = False
        
        if compute_sigmas:
            print(f"DEBUG latent shape: {latent_image['samples'].shape}")
            print(f"DEBUG latent ndim: {latent_image['samples'].ndim}")
            # Use negative indexing so it works for both 4D (SDXL) and 5D QWEN/Flux etc:
            width  = latent_image["samples"].shape[-1] * 8
            height = latent_image["samples"].shape[-2] * 8
            sigmas = self._get_sigmas(steps, width, height)
            # Force noise based options to their defaults.
            denoise = 1.0
            disable_noise = False
            start_at_step = 0
            end_at_step = 10000
            force_full_denoise = True
            log(f"{self.NODE_NAME}:📐 Computed FLUX sigmas for {width}x{height}.\nAll noise based customizations have been defaulted.")
        else:
            sigmas = None

        needs_clone = auraflow_shift > 0 or cfg_norm_strength > 0
        if needs_clone:
            model = model.clone()

            if auraflow_shift > 0:
                log(f"{self.NODE_NAME}:⚡Applying aura flow shift parameter of {auraflow_shift} ...")
                model = self._aura_flow_shift(model, auraflow_shift)
                log(f"{self.NODE_NAME}:✅ Shift parameter applied successfully")

            if cfg_norm_strength > 0:
                log(f"{self.NODE_NAME}:🎛️ Applying CFG Normalization strength of {cfg_norm_strength} ...")
                model = self._cfg_normalizer(model, cfg_norm_strength)
                log(f"{self.NODE_NAME}:✅ Normalization applied successfully")

        def _ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, disable_noise, start_step, last_step, force_full_denoise, sigmas=None):
            latent_image = latent["samples"]
            latent_image = c_sample.fix_empty_latent_channels(model, latent_image, latent.get("downscale_ratio_spacial", None))

            if disable_noise:
                noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
            else:
                batch_inds = latent["batch_index"] if "batch_index" in latent else None
                noise = c_sample.prepare_noise(latent_image, seed, batch_inds)

            noise_mask = None
            if "noise_mask" in latent:
                noise_mask = latent["noise_mask"]

            callback = latent_preview.prepare_callback(model, steps)
            disable_pbar = not c_utils.PROGRESS_BAR_ENABLED
            samples = c_sample.sample(
                callback           = callback,
                cfg                = cfg,
                denoise            = denoise,
                disable_noise      = disable_noise,
                disable_pbar       = disable_pbar,
                force_full_denoise = force_full_denoise,
                last_step          = last_step,
                latent_image       = latent_image,
                model              = model,
                negative           = negative,
                noise              = noise,
                noise_mask         = noise_mask,
                positive           = positive,
                sampler_name       = sampler_name,
                scheduler          = scheduler,
                seed               = seed,
                start_step         = start_step,
                steps              = steps,
                sigmas             = sigmas,
            )
            out = latent.copy()
            out.pop("downscale_ratio_spacial", None)
            out["samples"] = samples
            return out

        log(f"{self.NODE_NAME}:🚀 Generating, wait for it ...")
        latent_image = _ksampler(
            model,
            noise_seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise            = denoise,
            disable_noise      = disable_noise,
            start_step         = start_at_step,
            last_step          = end_at_step,
            force_full_denoise = force_full_denoise,
            sigmas             = sigmas
        )

        return (latent_image, )

# Flux helpers
def generalized_time_snr_shift(t, mu: float, sigma: float):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)

def get_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps