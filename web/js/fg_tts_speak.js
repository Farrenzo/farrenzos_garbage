const { api } = window.comfyAPI.api;
const { app } = window.comfyAPI.app;

// 🗑️ TTS Speak (FG_TTSSpeak): a status line on the node. "Speaking…" when it starts, then one
// line per spoken line (speech length, time, RTF, mode) from the node's "tts_status" ui output.
// The progress bar itself comes from the node's ProgressBar (sized to each line's estimated length).

const NODE_CLASSES = new Set(["FG_TTSSpeak"]);

document.head.append(Object.assign(document.createElement("style"), { textContent: `
.fg-tts-status { font: 11px sans-serif; color: var(--descrip-text, #999); padding: 2px 6px; white-space: pre-wrap; word-break: break-word; max-height: 140px; overflow-y: auto; }
.fg-tts-status.busy { color: var(--fg-color); }
` }));

function setupStatus(node) {
    const box = Object.assign(document.createElement("div"), { className: "fg-tts-status", textContent: "" });
    const widget = node.addDOMWidget("tts_status", "fg_tts_status", box, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => (box.textContent ? Math.min(140, box.scrollHeight + 4) : 0),
    });
    widget.serialize = false;

    const show = (text, busy) => {
        box.textContent = text;
        box.classList.toggle("busy", !!busy);
        requestAnimationFrame(() => {
            node.setSize([node.size[0], Math.max(node.size[1], node.computeSize()[1])]);
            node.setDirtyCanvas(true, true);
        });
    };

    const onExecuted = node.onExecuted;
    node.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        const lines = message?.tts_status;
        if (Array.isArray(lines)) show(lines.join("\n"), false);
    };
    node.fgTtsStatus = show;
}

// "Speaking…" as soon as ComfyUI starts this node
api.addEventListener("executing", ({ detail }) => {
    const id = detail?.node ?? detail;
    if (id == null) return;
    const node = app.graph?.getNodeById?.(Number(id)) ?? app.graph?.getNodeById?.(id);
    if (NODE_CLASSES.has(node?.comfyClass)) node.fgTtsStatus?.("Speaking… (progress bar above; the console shows frames)", true);
});

app.registerExtension({
    name: "FG.TTS.Speak.Status",

    async nodeCreated(node) {
        if (NODE_CLASSES.has(node.comfyClass)) setupStatus(node);
    },
});
