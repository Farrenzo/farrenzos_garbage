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

## Setting Up Ollama & WD14 Booru tagger

Because this node pack strongly believes in you downloading things for yourself you are going to have to download the WD_1.4 booru tagger model.  
Go to: [SmilingWolf Huggingface Repo](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/tree/main) & download:  

 - model.onnx
 - SHA256: 9e768793060c7939b277ccb382783e8670e8a042d29d77aa736be0c8cc898bfc
 - IMPORTANT: If you rename it, change the corresponding `env.json` entry.
 - Place the download in: `../custom_nodes/farrenzos_garbage/models/wd14_v3/model.onnx

Restart comfy & voila.

## Setting Up Dynamic LoRA Loader
This node works best if you go into your models/loras folder and fill out the prefilled `.json` file. Add trigger words and generate a few images for the preview. Use 128x128 jpg images and convert them into base 64. Contents should look like this:

```JSON
{
    "some_lora_name_at_top_level_of_lora_folder.safetensors":{
        "trigger_words": "trigger_word_1, trigger_word_2, trigger_word_3, etc this is a string or long sentence",
        "preview_image": "data:image/png;base64,REALLY_REALLY_REALLY_LONG_STRING"
    },
    "some_folder_name\\some_lora_name.safetensors":{
        "trigger_words": "trigger_word_1, trigger_word_2, trigger_word_3, etc this is a string or long sentence",
        "preview_image": "data:image/png;base64,REALLY_REALLY_REALLY_LONG_STRING"
    },
    "some_other_folder_name\\some_other_lora_name.safetensors":{
        "trigger_words": "trigger_word_1, trigger_word_2, trigger_word_3, etc this is a string or long sentence",
        "preview_image": "data:image/jpg;base64,REALLY_REALLY_REALLY_LONG_STRING"
    }
}
```
Tip: you can use this same file to save lora info like site downloaded, SHA256, etc. The node only cares about the two top keys trigger words & preview image.

### Notice
1. The `aiohttp` call in the beginning is only so that javascript may communicate with the LoRA loader node. Nothing will be downloaded on your behalf.

