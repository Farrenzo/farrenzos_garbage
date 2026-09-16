"""
FG_StepInspect — watch the model's guess evolve, step by step.

At every step the sampler asks the model for a COMPLETE prediction of the
finished image (`x0`), then travels only part of the way toward it. ComfyUI
hands that prediction to the sampling callback and normally throws it away after
drawing a preview thumbnail. This node keeps them all and decodes them into an
IMAGE batch.

What that buys you: you can see exactly which step your layout locked, which
step the palette committed, and therefore where it is worth interrupting to
inject an edit.

Decode `x0`, never `x`. At high sigma `x` is mostly noise and tells you nothing;
`x0` is the model's actual opinion and is legible from step 1.

INTERRUPT AND RESUME (no custom sampler needed, stock nodes do it):

  1. FG_StepInspect, end_step = k, return_with_leftover_noise = True
  2. VAEDecode -> edit the pixels (paint the hair red, FG_HueCorrect, whatever)
  3. VAEEncode
  4. KSamplerAdvanced: add_noise = enable, start_step = k, end_step = steps,
     SAME seed and steps. Comfy re-noises your edited latent to sigma[k], which
     is what makes the remaining steps treat it as a work in progress rather
     than a finished image.

  Small k = your edit is a suggestion the model rebuilds around.
  Large k = your edit is nearly a hard paste.
"""

import comfy.sample
import comfy.samplers
import comfy.utils
import torch


class FG_StepInspect:
    """Sample, and return every step's x0 prediction as an image batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "start_step": ("INT", {
                    "default": 0, "min": 0, "max": 10000,
                    "tooltip": "Resume from here. Use with add_noise=True and the "
                               "SAME seed/steps to continue an interrupted run."}),
                "end_step": ("INT", {
                    "default": 10000, "min": 0, "max": 10000,
                    "tooltip": "Stop here. Set leftover_noise=True when stopping "
                               "early, or the result is forced to a clean image and "
                               "cannot be resumed."}),
                "add_noise": ("BOOLEAN", {"default": True}),
                "leftover_noise": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Leave the latent partly noisy so another sampler can "
                               "pick it up. Required for the interrupt-and-resume "
                               "workflow."}),
            },
            "optional": {
                "vae": ("VAE", {
                    "tooltip": "Needed for the trajectory output. Use a TAESD VAE "
                               "here — a full VAE decode per step is slower than the "
                               "sampling itself."}),
                "decode_every": ("INT", {
                    "default": 1, "min": 1, "max": 50,
                    "tooltip": "Decode every Nth step. Raise it if previewing costs "
                               "more than you want."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("latent", "trajectory", "report")
    FUNCTION = "sample"
    CATEGORY = "Farrenzo's Garbage/Sampling"
    DESCRIPTION = "Sampler that returns every step's x0 prediction as images."

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               sampler_name, scheduler, start_step, end_step, add_noise,
               leftover_noise, vae=None, decode_every=1, denoise=1.0):
        latent = latent_image["samples"]
        latent = comfy.sample.fix_empty_latent_channels(model, latent)

        if add_noise:
            batch_inds = latent_image.get("batch_index")
            noise = comfy.sample.prepare_noise(latent, seed, batch_inds)
        else:
            noise = torch.zeros(latent.size(), dtype=latent.dtype,
                                layout=latent.layout, device="cpu")

        captured = []

        def callback(step, x0, x, total_steps):
            if step % decode_every == 0 or step == total_steps - 1:
                captured.append((step, x0.detach().to("cpu", torch.float32)))

        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent,
            denoise=denoise,
            disable_noise=not add_noise,
            start_step=start_step,
            last_step=end_step,
            force_full_denoise=not leftover_noise,
            noise_mask=latent_image.get("noise_mask"),
            callback=callback,
            seed=seed,
        )

        # Sigma schedule, so you can see where the big jumps are. Early steps cover
        # the most ground, which is why layout commits so soon.
        sig_line = ""
        try:
            ms = model.get_model_object("model_sampling")
            sigmas = comfy.samplers.calculate_sigmas(ms, scheduler, steps).tolist()
            shown = [f"{i}:{s:.3f}" for i, s in enumerate(sigmas)]
            sig_line = "sigmas    : " + "  ".join(shown)
        except Exception as e:
            sig_line = f"sigmas    : unavailable ({e})"

        traj = torch.zeros(1, 8, 8, 3)
        if vae is not None and captured:
            try:
                proc = model.get_model_object("process_latent_out")
            except Exception:
                proc = lambda t: t

            frames = []
            for _, x0 in captured:
                img = vae.decode(proc(x0))
                if img.ndim == 5:
                    img = img.reshape(-1, *img.shape[-3:])
                frames.append(img[:1, ..., :3].cpu())
            traj = torch.cat(frames, dim=0).clamp(0, 1)

        last = end_step if end_step < steps else steps
        report = "\n".join([
            f"steps     : {start_step} -> {last} of {steps}",
            f"noise     : add={add_noise}  leftover={leftover_noise}",
            f"captured  : {len(captured)} x0 predictions "
            f"(every {decode_every} step{'s' if decode_every != 1 else ''})",
            f"trajectory: {tuple(traj.shape)}"
            + ("" if vae is not None else "   <- connect a VAE (TAESD) to get images"),
            sig_line,
            "",
            "Read the trajectory left to right. The frame where layout stops moving "
            "is where composition locked; the frame where colour stops shifting is "
            "where palette locked. Inject BEFORE those and the model rebuilds around "
            "your edit. Inject after and it can only accept or fight it.",
        ])
        print("[FG_StepInspect]\n" + report)

        if not leftover_noise and end_step < steps:
            print("[FG_StepInspect] WARNING: stopped early with leftover_noise=False. "
                  "The latent was forced to a clean image and will not resume "
                  "correctly. Turn leftover_noise on.")

        out = latent_image.copy()
        out["samples"] = samples
        return (out, traj, report)
