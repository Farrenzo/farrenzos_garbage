"""
"""
import math
import torch
import comfy.utils
import node_helpers
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict

global_description = """
Enhanced CLIP Text Encoder. Encodes text prompts using
a CLIP model into an embedding that can be used to guide
the diffusion model towards generating specific images.

Many SDXL models will use a negative prompt. However, newer
models appear to have better (not perfect) prompt adherence.
Hence a negative prompt is not always required.
"""
global_instruction_content = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."


class FG_CLIPTextEncode(ComfyNodeABC):

    def __init__(self):
        self.NODE_NAME = "Enhanced CLIP Text Encode"

    @classmethod
    def INPUT_TYPES(self) -> InputTypeDict:
        return {
            "required": {
                "clip"            : (IO.CLIP, {"tooltip": "📒 Clip: The CLIP model used for encoding the text."}),
                "prompt_option"   : (IO.BOOLEAN, {"default": False, "label_on": "Negative prompt is not required.", "label_off": "Enter a negative prompt."}),
                "vl_based_clip"   : (IO.BOOLEAN, {"default": False, "label_on": "Visible VL Textbox.", "label_off": "Hidden VL Textbox.", "tooltip": "Typically used for CLIPS that are good at translating images into conditioning."}),
                "positive_prompt" : (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "✅ You WANT to see."}),
            },
            "optional": {
                "vae": (IO.VAE, {"tooltip": "🔣 VAE: Used to encode the reference images into latent the model understands."}),
                "image1":(IO.IMAGE, {"tooltip": "🖼️ Picture 1 — Reference Image 1, used for conditional generation and latent space encoding."}),
                "image2":(IO.IMAGE, {"tooltip": "🖼️ Picture 2 — Reference Image 2, used for conditional generation and latent space encoding."}),
                "image3":(IO.IMAGE, {"tooltip": "🖼️ Picture 3 — Reference Image 3, used for conditional generation and latent space encoding."}),
                "negative_prompt": (
                    IO.STRING, {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "❌ You DON'T want to see.",
                        "placeholder": "Enter the negative prompt"
                    }
                ),
                "vl_instruction": (
                    "STRING", {
                        "multiline": True, 
                        "default": global_instruction_content, 
                        "placeholder": global_instruction_content,
                        "tooltip": "📝 System instructions used to guide the editing of reference images."
                    }
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("Positive", "Negative")
    # "Conditioning that contains embedded text used to guide the diffusion model."
    OUTPUT_TOOLTIPS = ("Attach this to the + side of the KSampler", "Attach this to the - side of the KSampler",)
    FUNCTION = "encode"

    CATEGORY = "Farrenzo's Garbage/Conditioning"
    DESCRIPTION = global_description
    SEARCH_ALIASES = ["text", "prompt", "text prompt", "positive prompt", "negative prompt", "encode text", "text encoder", "encode prompt"]

    def encode(self,
        clip,
        positive_prompt,
        vae    = None,
        image1 = None,
        image2 = None,
        image3 = None,
        negative_prompt="",
        vl_based_clip = False,
        prompt_option = False,
        vl_instruction = global_instruction_content # Vision Language Instructions for the CLIP.
    ):
        vl_based_clip = vl_based_clip
        def _advanced_encode(prompt):
            images = [image1, image2, image3]
            images_vl   = [] # The scaled down images will be popped in here if any are provided.
            ref_latents = [] # The latent images will be popped in here if a VAE is passed in.
            llama_template = f"<|im_start|>system\n{vl_instruction}<|im_end|>\n<|im_start|>user\n{{}} <|im_end|>\n<|im_start|>assistant\n"
            image_prompt = ""

            for i, image in enumerate(images):
                if image is not None:
                    image_prompt += f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>\n"
                    samples = image.movedim(-1, 1)
                    total = int(384 * 384)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by)
                    height = round(samples.shape[2] * scale_by)
                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    images_vl.append(s.movedim(1, -1))

                    if vae is not None:
                        # Keep original resolution, just round to nearest multiple of 8
                        width = (samples.shape[3] + 7) // 8 * 8
                        height = (samples.shape[2] + 7) // 8 * 8
                        s = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
                        ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

            if image_prompt == "":
                tokens = clip.tokenize(prompt)
            else:
                tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)

            conditioning = clip.encode_from_tokens_scheduled(tokens)
            if len(ref_latents) > 0:
                conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
            return conditioning
    
        def _zero_out(conditioning):
            c = []
            for t in conditioning:
                d = t[1].copy()
                pooled_output = d.get("pooled_output", None)
                if pooled_output is not None:
                    d["pooled_output"] = torch.zeros_like(pooled_output)
                conditioning_lyrics = d.get("conditioning_lyrics", None)
                if conditioning_lyrics is not None:
                    d["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
                n = [torch.zeros_like(t[0]), d]
                c.append(n)
            return c

        positive_conditioning = _advanced_encode(positive_prompt)
        if prompt_option:
            negative_conditioning = _zero_out(positive_conditioning)
        else:
            negative_conditioning = _advanced_encode(negative_prompt)

        return (positive_conditioning, negative_conditioning, )

