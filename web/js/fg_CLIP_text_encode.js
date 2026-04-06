const { app } = window.comfyAPI.app;

const WIDGET_HEIGHT = 55;

function hideWidget(node, widget) {
    if (!widget || widget.hidden) return;
    widget.origType = widget.type;
    widget.hidden = true;
    widget.type = "converted-widget";
    node.setSize([node.size[0], node.size[1] - WIDGET_HEIGHT]);
}

function showWidget(node, widget) {
    if (!widget || !widget.hidden) return;
    widget.type = widget.origType || widget.type;
    widget.hidden = false;
    node.setSize([node.size[0], node.size[1] + WIDGET_HEIGHT]);
}

function setupNode(node) {
    const vlBasedClipWidget      = node.widgets.find(w => w.name === "vl_based_clip");
    const onlyPositiveWidget     = node.widgets.find(w => w.name === "prompt_option");
    const negativePromptWidget   = node.widgets.find(w => w.name === "negative_prompt");
    const vlInstructionWidget    = node.widgets.find(w => w.name === "vl_instruction");

    if (!vlBasedClipWidget || !onlyPositiveWidget) return;

    function updateVisibility() {
        if (onlyPositiveWidget.value === true) {
            hideWidget(node, negativePromptWidget);
        } else {
            showWidget(node, negativePromptWidget);
        }

        if (vlBasedClipWidget.value === true) {
            showWidget(node, vlInstructionWidget);
        } else {
            hideWidget(node, vlInstructionWidget);
        }
    }

    updateVisibility();

    const origVlCallback = vlBasedClipWidget.callback;
    vlBasedClipWidget.callback = function (value) {
        if (origVlCallback) origVlCallback.call(this, value);
        updateVisibility();
    };

    const origPosCallback = onlyPositiveWidget.callback;
    onlyPositiveWidget.callback = function (value) {
        if (origPosCallback) origPosCallback.call(this, value);
        updateVisibility();
    };
}

app.registerExtension({
    name: "Farrenzo.DynamicCLIPTextEncode",

    async nodeCreated(node) {
        if (node.comfyClass !== "FG_CLIPTextEncode") return;
        setupNode(node);
    },

    async loadedGraphNode(node) {
        if (node.comfyClass !== "FG_CLIPTextEncode") return;
        setTimeout(() => setupNode(node), 100);
    }
});
