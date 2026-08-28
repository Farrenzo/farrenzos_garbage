// Uniform colors for all garbage nodes.
const { api } = window.comfyAPI.api;
const { app } = window.comfyAPI.app;

app.registerExtension({
    name: "FG_Node_Colors",

    async loadedGraphNode(node) {
        const garbage_nodes = [
            "FG_Advanced_KSampler",
            "FG_ApplyControlNet",
            "FG_BoxFillwCoordinates",
            "FG_CLIPTextEncode",
            "FG_CombinedImageTagger",
            "FG_CustomVAELoader",
            "FG_DynamicLoraLoader",
            "FG_EmptyLatent",
            "FG_ImageScaler",
            "FG_KSampler",
            "FG_LABColorTransfer",
            "FG_LoadImage",
            "FG_Minimum_Maximum",
            "FG_ModelImageScaler",
            "FG_ModelReferenceLatentMethod",
            "FG_PurgeMemory",
            "FG_SaveImage",
            "FG_SaveVideo",
            "FG_SendTelegramNotification",
            "FG_ShowText",
            "FG_WD14Tagger",
            "FG_MiniMaxH3_Conditioner",
            "FG_XPUGuard",
            "FG_UnifiedModelsLoader",

            "FG_XPUGuard",
            "AnimaConditioningRegion",
            "ApplyAnimaRegionalConditioningPatch",
            "AnimaLLLiteApply",

            "AnimaIPAdapterLoader",
            "AnimaIPAdapterApply",
            "AnimaSiglipeEncodeImage",
            "AnimaImageEmbLoader",

            "MultiLatentComposite",
            "MultiAreaConditioning",
            "ConditioningUpscale",
            "ConditioningStretch",

            "ConditioningKrea2Rebalance",

            "SCAIL2EasyConfig",
            "SCAIL2AutoVideo",
            "SCAIL2RunInfo",

            "MiniMaxH3AutoChainMotionContext",
            "MiniMaxH3AutoChainMotionContextTrim",
            "MiniMaxH3AutoChainSaveLatent",
            "MiniMaxH3AutoChainLoadLatent",
            "MiniMaxH3AutoChainAudio",
            "MiniMaxH3AutoChain",
            "MiniMaxH3AutoChainFrameReference",
        ];
        if (!garbage_nodes.includes(node.comfyClass)) {
            return;
        }
        node.color   = "#222233";
        node.bgcolor = "#0c161bd8";
    },
});

