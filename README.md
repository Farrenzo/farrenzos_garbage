# 🗑️ Farrenzo's Garbage Nodes

Some custom ComfyUI nodes. These include:

1. A better save image node. Names files with UTC. Best used if you organize your images by folder
2. A LoRA loader node that you simply stack them one after another.
3. An image tagger that incorporates both booru style and natural language descriptor.
4. Better CLIP Text Encode for positive and negative and image prompt conditioning.

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

## Setting Up Dynamic LoRA Loader

After the initial run of comfy with this node pack, inside your LoRA's folder you will find a folder called `.lora_previews`. You can move this folder to any LoRA folder that comfy recognizes. If you have a central location for all your models, and they are properly listed in comfy's `extra_model_paths.yaml` file, the folder will be visible. Inside `.lora_previews` you will find a prefilled `_fg_dynamic_lora_loader.json` index file. Edit that index file with trigger words to your hearts content. Generate a few images for the preview. Use square 512*512 webp images and save them in the previews folder under the **exact** same name as the LoRA. So if your LoRA is called `something_v1.safetensors` the preview image should be called `something_v1.jpg/webp/png/etc`. Webp is recommended due to it's tiny size.

```JSON
{
    "some_lora_name_at_top_level_of_lora_folder.safetensors":{
        "trigger_words": "trigger_word_1, trigger_word_2, trigger_word_3, long sentence"
    },
    "some_folder_name\\some_lora_name.safetensors":{
        "trigger_words": "trigger_word_1, trigger_word_2, trigger_word_3, something descriptive"
    },
    "some_other_folder_name\\some_other_lora_name.safetensors":{
        "trigger_words": "trigger_word_1, trigger_word_2, trigger_word_3, I've ran out of creativity"
    }
}
```
Tip: you can use this same file to save lora info like site downloaded, SHA256, etc. The node only cares about the trigger word. Storing LoRA sha256 makes it easy to search for later and compare if you have that file. No restart required. It's all javascript so a refresh will show you changes instantly.

### Notice
1. The `aiohttp` call in the beginning is only so that javascript may communicate with the LoRA loader node. Nothing will be downloaded on your behalf.

