# 🗑️ Farrenzo's Garbage Nodes

Some custom ComfyUI nodes. These include:

1. A better save image node. Names files with UTC. Best used if you organize your images by folder
2. A LoRA loader node that you simply stack them one after another.
3. An image tagger.
4. Better CLIP Text Encode for positive, negative & image prompt conditioning.

Complete list below. The **gear** (⚙️) icone means experimental. The rest are not only ready to use but can be used immediately.

```
    🗑️ Advanced KSampler
    🗑️ Apply Advanced ControlNet
    🗑️ Advanced Edge Detector
    🗑️ Enhanced CLIP Text Encode
    🗑️ Combined Image Tagger
    🗑️ Multi-LoRA Loader
    🗑️ Advanced Empty Latent
    🗑️ Image Scaler
    🗑️ KSampler for Qwen Image Edit
    🗑️ LAB Color Transfer
    🗑️ Latent Transfer (VAE to VAE)
    🗑️ Load Image
    🗑️ MiniMax H3 Conditioner
    🗑️ Edit Model Reference Method
    🗑️ Purge Memory
    🗑️ Anima Role Reference KV Patch
    🗑️ Save Image
    🗑️ Save Video
    🗑️ Send Telegram Notification
    🗑️ Show Text
    🗑️ Unified Models Loader
    🗑️ WD14 Tagger (Booru Tags)
    🗑️ Resource Manager
    🗑️ XPU Guard (Device Health)
    🗑️ YuE2 XPU Accelerator
    🗑️ MiniMaxMusicH3 XPU Accelerator
    🗑️ Step Inspect (x0 trajectory)
    ⚙️ Anima Apply ControlNet-LLLite
    🗑️ Anima IP-Adapter Apply
    🗑️ Anima IP-Adapter Loader
    🗑️ Anima IP Adapter Visualizer
    🗑️ Anima SigLIP2 Encode Image
    ⚙️ LLLite Cond Preview
    🗑️ Anima Regional Prompt
    🗑️ Anima Apply Cosmos Reference Latent
    🗑️ Krea2 Edit Model Patch
    🎛️ Krea 2 Conditioning Control
    ⚙️ Mask Backdrop
    🗑️ Crop Factor Padding
    🔧 Mask From RGB/CMY/BW
    🗑️ Coordinate Box Fill
    ✂️ Auto Crop & Clip
    🪡 Auto Crop & Clip ReStitcher
    🎲 Noisy Seeder
    🎲 Seed Variance Enhancer
    🗑️ Color Blender
    🗑️ Hue Correct (LAB)
    🗑️ Image Forensics
    🗑️ Color Picker
    🗑️ Linework Composite
    🗑️ Color Map (for Bubble Tint)
    🗑️ Bubble Tint (colour lettering)
    ⚙️ H3 Auto Chain Motion Context
    ⚙️ H3 Auto Chain Motion Context Trim
    ⚙️ H3 Auto Chain Save Latent
    ⚙️ H3 Auto Chain Load Latent
    ⚙️ H3 Auto Chain Audio
    ⚙️ H3 Auto Chain + Stitch
    ⚙️ H3 Auto Chain Frame Reference
    ⚙️ Conditioning Debug
    ⚙️ Conditioning Upscale
    ⚙️ Conditioning Stretch
    ⚙️ Multi Latent Composite
    ⚙️ Multi Area Conditioning
    ⚙️ SCAIL-2 Easy Config
    ⚙️ SCAIL-2 Auto Video (loop)
    ⚙️ SCAIL-2 Run Info
    🗑️ Save Conditioning / Latent To Disk
    🗑️ Load Conditioning / Latent From Disk
    🗑️ Any Switcher
    🗑️ Int / Float Out
    🗑️ Minimum + Maximum
    ⚙️ Viggle Remake Noise
    ⚙️ Viggle Remake Conditioning
    ⚙️ Viggle Remake Keyframe Until
    🎨 Inpaint Canvas
    🎨 Inpaint Canvas Stitch
    🎨 Inpaint Canvas Load Ref
    🎨 Inpaint Canvas Mask Out
    🎨 Inpaint Canvas Object Map
    🎨 Inpaint Canvas Text Out
```

Folder structure:
---
```
... /custom_nodes/farrenzos_garbage/
│
├───__init__.py
├───requirements.txt
│
├───js/
│   ├───dynamic_lora.js
│   └───show_text.js
│
├───css/
│   └───custom.css
│
├───nodes/
│   ├───__init__.py
│   ├───combined_image_tagger.py
│   ├───dynamic_lora_loader.py
│   ├───save_image_clean.py
│   └───show_text.py
│
└───workflows/
    └───tagger.json
```

# Installation
They should all work right out of the box with no pip setups required. 

- Git clone to `..\custom_nodes\` folder.
- After initial run, a blank `env.json` file will be created. It should look like this inside:

```json
{
    "TELEGRAM_CHAT_ID": null,
    "TELEGRAM_PRIVATE_API": null,
    "WD_14_TAGGER": {
        "directory": "wd14_v3",
        "tagging_models":{
            "eva02-large": {
                "model": "model.onnx",
                "csv": "wd-eva02-large-tagger-v3.csv"
            }
        }
    },
    "GOOGLE_SIGLIP2": {
        "hf_repo":"google/siglip2-base-patch16-512",
        "directory": "siglip2"
    }
}
```
- If the above file does not exist, just create it. It's a text file.

## Setting Up TeleGram

### Creating a Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Start a chat and send `/newbot`
3. Follow the instructions to create your bot
4. Save the bot token provided by BotFather to the `"TELEGRAM_PRIVATE_API": something_here_that_you_got`. 

### Getting Your Chat ID

1. Start a chat with your bot
2. Send any message to your bot
3. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
4. Look for the `chat.id` field in the response
5. Save the chat id to the `"TELEGRAM_CHAT_ID": the_chat_id`. 

## Setting Up required models:
This node pack firmly believes in you downloading things for yourself. Know what is running in your systems. You are responsible for downloading the booru tagger and siglip base models.

***WD14 Booru tagger***: Go to: [SmilingWolf Huggingface Repo](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/tree/main) & download: 

 - model.onnx
 - **SHA256**: 9e768793060c7939b277ccb382783e8670e8a042d29d77aa736be0c8cc898bfc
 - ***Important***: If you rename it, change the corresponding `env.json` entry.
 - Place the download in: `../custom_nodes/farrenzos_garbage/models/wd14_v3/model.onnx`

***Google Siglip***: Go to: [SigLIP 2 Base](https://huggingface.co/google/siglip2-base-patch16-512/main) & download:

 - model.safetensors
 - **SHA256**: fe0e601c625e69eed8e73500d39e9b6164403fe03db8048e87913c3cefbbb3fe
 - Place the download in: `../custom_nodes/farrenzos_garbage/models/siglip2/model.safetensors`

Restart comfy & voila.

## Resource Library (LoRA trigger words, previews, recommended settings)

Model info lives in a SQLite library keyed by each file's SHA256, so renaming or moving a model keeps its info and deleted models stay on record as missing.

1. Set `"RESOURCE_LIBRARY_DIR"` in `_env.json` to the folder that should hold the library, e.g. your central models folder (`"X:/xModels/Models"`). It defaults to ComfyUI's `models` folder.
2. Run the **🗑️ Resource Manager** node (connect its report to Show Text). It hashes every model ComfyUI can see, including `extra_model_paths.yaml` folders. Only new or changed files are hashed on later runs.
3. Right-click any model loader › **📝 Edit info** to set base model, source URL, trigger words, notes, recommended settings and a preview image (click, drop or paste). The Multi-LoRA Loader shows a submenu with one item per LoRA.

Previews are saved in `<library>/.previews/`, mirroring the model folders with readable names, e.g. `.previews/loras/Anima/Anima_Style_Meteor.webp`. Square 512×512 webp is recommended. The Multi-LoRA Loader shows previews and trigger words; checkpoint and diffusion model loaders show the preview and recommended settings under their widgets.

### Notice
1. The `/fg/library` routes only let the browser read and edit your local resource library. Nothing is downloaded or sent anywhere.

