const { app } = window.comfyAPI.app;

/*
 * Colour swatch widgets for Farrenzo's Garbage.
 *
 * Why the obvious approach fails: ComfyUI nodes are painted onto a <canvas> by
 * LiteGraph. Widgets are draw calls, not DOM elements, so there is nothing to
 * attach <input type="color"> to, and anything appended over the canvas has its
 * pointer events eaten by LiteGraph's own handlers before the native dialog can
 * open.
 *
 * What works:
 *   1. Keep the original STRING widget. Do NOT push a new one — widgets_values
 *      serialises by index, so adding a widget silently breaks every saved
 *      workflow that used the node.
 *   2. Override that widget's draw() and mouse() in place. Same widget, same
 *      index, same value, different appearance and click behaviour.
 *   3. On click, append a detached <input type="color"> to document.body and
 *      call showPicker() SYNCHRONOUSLY inside the handler, so the browser still
 *      counts it as a user gesture. Remove it when done.
 */

const HEX = /^#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})$/;

function normalise(v, fallback = "#000000") {
    if (typeof v !== "string") return fallback;
    const m = v.trim().match(HEX);
    if (!m) return fallback;
    let h = m[1];
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    return "#" + h.toLowerCase();
}

function isColourWidget(name) {
    const n = String(name || "").toLowerCase();
    return n === "color" || n === "colour" ||
           n.endsWith("_color") || n.endsWith("_colour") || n.endsWith("_hex");
}

function openNativePicker(initial, onInput, onDone) {
    const input = document.createElement("input");
    input.type = "color";
    input.value = normalise(initial);
    // Off-screen but attached: showPicker() throws on a disconnected element.
    Object.assign(input.style, {
        position: "fixed", left: "-9999px", top: "0px",
        width: "1px", height: "1px", opacity: "0", pointerEvents: "none",
    });
    document.body.appendChild(input);

    let finished = false;
    const cleanup = () => {
        if (finished) return;
        finished = true;
        setTimeout(() => input.remove(), 0);
    };

    input.addEventListener("input", (e) => onInput(e.target.value));
    input.addEventListener("change", (e) => { onInput(e.target.value); onDone && onDone(); cleanup(); });
    // Some browsers fire neither if the dialog is dismissed; reclaim on blur.
    window.addEventListener("focus", () => setTimeout(cleanup, 250), { once: true });

    try {
        if (typeof input.showPicker === "function") input.showPicker();
        else input.click();
    } catch (err) {
        input.click();
    }
}

function upgrade(node, widget) {
    if (widget.__fgSwatch) return;
    widget.__fgSwatch = true;

    widget.draw = function (ctx, n, width, y, H) {
        const margin = 15;
        const hex = normalise(this.value, "#000000");
        const r = H * 0.35;

        ctx.save();
        // pill background, matching LiteGraph's own widget styling
        ctx.fillStyle = "#222";
        ctx.strokeStyle = "#444";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(margin, y, width - margin * 2, H, r);
        ctx.fill();
        ctx.stroke();

        // the swatch itself
        const sw = H - 8;
        ctx.fillStyle = hex;
        ctx.strokeStyle = "#000";
        ctx.beginPath();
        ctx.roundRect(margin + 5, y + 4, sw, sw, 3);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = "#999";
        ctx.font = "12px Arial";
        ctx.textAlign = "left";
        ctx.fillText(this.name, margin + sw + 12, y + H * 0.7);

        ctx.fillStyle = "#ddd";
        ctx.textAlign = "right";
        ctx.fillText(hex, width - margin - 8, y + H * 0.7);
        ctx.restore();
    };

    widget.mouse = function (event, pos, n) {
        if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
        const self = this;
        openNativePicker(
            self.value,
            (v) => {
                self.value = normalise(v);
                if (self.callback) self.callback(self.value);
                n.setDirtyCanvas(true, true);
            },
            () => { if (n.graph) n.graph.setDirtyCanvas(true, true); }
        );
        return true;   // consume the click so LiteGraph doesn't open its text prompt
    };

    widget.computeSize = function (width) {
        return [width, 24];
    };
}

app.registerExtension({
    name: "farrenzos_garbage.colour_swatch",
    async nodeCreated(node) {
        if (!node.widgets) return;
        for (const w of node.widgets) {
            if (w.type === "text" || w.type === "string" || w.type === "customtext") {
                if (isColourWidget(w.name)) upgrade(node, w);
            }
        }
    },
});