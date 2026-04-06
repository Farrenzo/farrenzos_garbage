// Uniform colors for all garbage nodes.
const { api } = window.comfyAPI.api;
const { app } = window.comfyAPI.app;

app.registerExtension({
    name: "FG_Node_Colors",

    async loadedGraphNode(node) {
        const garbage_nodes = [
            "FG_Advanced_KSampler",
            "FG_ApplyControlNet",
            "FG_EmptyLatent",
            "FG_BoxFillwCoordinates",
            "FG_CLIPTextEncode",
            "FG_CombinedImageTagger",
            "FG_CustomVAELoader",
            "FG_DynamicLoraLoader",
            "FG_Florence2Captioner",
            "FG_ImageScaler",
            "FG_ModelImageScaler",
            "FG_KSampler",
            "FG_LABColorTransfer",
            "FG_LoadImage",
            "FG_Minimum_Maximum",
            "FG_PurgeMemory",
            "FG_SaveImage",
            "FG_SendTelegramNotification",
            "FG_ShowText",
            "FG_WD14Tagger",
            "FG_ModelReferenceLatentMethod"
        ];
        if (!garbage_nodes.includes(node.comfyClass)) {
            return;
        }
        node.color   = "#222233";
        node.bgcolor = "#0c161bd8";
    },
});