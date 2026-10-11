import { app } from "../../../scripts/app.js";

// Show the bbox_model and bbox_threshold widgets only while bbox_detect is on.
app.registerExtension({
    name: "FG.LineworkComposite",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FG_LineworkComposite") return;

        function sync() {
            const toggle = this.widgets?.find((w) => w.name === "bbox_detect");
            if (!toggle) return;
            for (const w of this.widgets) {
                if (w.name === "bbox_model" || w.name === "bbox_threshold") w.hidden = !toggle.value;
            }
            this.setSize([this.size[0], this.computeSize()[1]]);
            this.setDirtyCanvas(true, true);
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const toggle = this.widgets?.find((w) => w.name === "bbox_detect");
            if (toggle) {
                const callback = toggle.callback;
                toggle.callback = (...args) => {
                    const cr = callback?.apply(toggle, args);
                    sync.call(this);
                    return cr;
                };
            }
            sync.call(this);
            return r;
        };

        // Saved workflows set widget values after creation, without the callback.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            sync.call(this);
            return r;
        };
    },
});
