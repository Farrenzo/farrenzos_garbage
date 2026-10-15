// Uniform colors for all garbage nodes.
const { api } = window.comfyAPI.api;
const { app } = window.comfyAPI.app;

app.registerExtension({
    name: "FG_Node_Colors",

    async loadedGraphNode(node) {
        const garbage_nodes = [
            "FG_Advanced_KSampler",
            "FG_AnimaEasyControlPatch",
            "FG_ApplyControlNet",
            "FG_ApplyCosmosReferenceLatent",
            "FG_BoxFillwCoordinates",
            "FG_BubbleTint",
            "FG_CannyDetect",
            "FG_CLIPTextEncode",
            "FG_CombinedImageTagger",
            "FG_CropFactorPadding",
            "FG_CustomVAELoader",
            "FG_DynamicLoraLoader",
            "FG_EmptyLatent",
            "FG_ImageScaler",
            "FG_IntFloatOut",
            "FG_KSampler",
            "FG_LABColorTransfer",
            "FG_LatentTransfer",
            "FG_LineworkComposite",
            "FG_LoadConditioning",
            "FG_LoadImage",
            "FG_MaskFromRGBCMYBW+",
            "FG_MiniMaxH3_Conditioner",
            "FG_MiniMaxMusicXPUAccel",
            "FG_Minimum_Maximum",
            "FG_ModelImageScaler",
            "FG_ModelReferenceLatentMethod",
            "FG_NoisySeeder",
            "FG_PurgeMemory",
            "FG_ResourceManager",
            "FG_SaveConditioning",
            "FG_SaveImage",
            "FG_SaveVideo",
            "FG_SendTelegramNotification",
            "FG_ShowText",
            "FG_UnifiedModelsLoader",
            "FG_WD14Tagger",
            "FG_XPUGuard",
            "FG_YuE2XPUAccel",

            "FG_AnimaLLLiteApply",
            "FG_LLLiteCondPreview",
            "ApplyAnimaRegionalConditioningPatch",
            "AnimaIPAdapterLoader",
            "AnimaIPAdapterApply",
            "AnimaSiglipeEncodeImage",
            "AnimaImageEmbLoader",
            "AnimaIPAdapterVisualize",
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
            
            "UnetLoaderGGUF",
            "CLIPLoaderGGUF",
            "DualCLIPLoaderGGUF",
            "TripleCLIPLoaderGGUF",
            "QuadrupleCLIPLoaderGGUF",
            "UnetLoaderGGUFAdvanced",
            "FG_HueCorrect",
            "FG_ImageForensics",
            "FG_StepInspect",
            "FG_MaskBackdrop",
            "FG_ColorPicker",
            "FG_ColorMap",
            "FG_SeedVarianceEnhancer",
        ];
        if (!garbage_nodes.includes(node.comfyClass)) {
            return;
        }
        node.color   = "#222233";
        node.bgcolor = "#0c161bd8";
    },
});

