const { api } = window.comfyAPI.api;
const { app } = window.comfyAPI.app;

let loraList = [];
let loraIndex = {};

async function fetchLoraList() {
    try {
        const resp = await api.fetchApi("/object_info");
        const info = await resp.json();
        for (const nodeName in info) {
            const node = info[nodeName];
            if (node.input?.required?.lora_name) {
                loraList = node.input.required.lora_name[0];
                break;
            }
        }
    } catch (e) {
        console.error("Failed to fetch LoRA list:", e);
    }
}

async function fetchLoraIndex() {
    try {
        const resp = await api.fetchApi("/fg/lora_index");
        loraIndex = await resp.json();
    } catch (e) {
        console.error("Failed to fetch LoRA index:", e);
    }
}

fetchLoraList();
fetchLoraIndex();

function getLoraInfo(loraName) {
    if (!loraName) return null;
    if (loraIndex[loraName]) return loraIndex[loraName];
    const alt = loraName.replace(/\//g, "\\");
    if (loraIndex[alt]) return loraIndex[alt];
    const alt2 = loraName.replace(/\\/g, "/");
    if (loraIndex[alt2]) return loraIndex[alt2];
    return null;
}

function truncate(str, max) {
    return str.length > max ? str.substring(0, max - 1) + "…" : str;
}

// ─────────────────────────────────────────────────────────────────
//  Shifted widget drawing — pushes widgets right to make room
//  for the preview image column on the left.
//
//  LiteGraph calls widget.draw(ctx, node, widget_width, y, H)
//  if the function exists; otherwise it uses its built-in renderer.
//  By providing our own, we control where the widget paints.
// ─────────────────────────────────────────────────────────────────

const IMG_COL_RATIO = 0.25; // left column takes 25% of node width

function getImgColWidth(node) {
    return Math.floor(node.size[0] * IMG_COL_RATIO);
}

/**
 * Factory: returns a draw function for the given widget type
 * that renders the widget shifted to the right by the image column width.
 */
function createShiftedDraw(widgetType) {
    return function (ctx, node, widget_width, y, H) {
        this.last_y = y;

        const shift = getImgColWidth(node);
        const margin = 15;
        const x = shift + margin;
        const bw = widget_width - shift - margin * 2;
        if (bw <= 0) return;

        const bgColor    = LiteGraph.WIDGET_BGCOLOR              || "#232323";
        const outColor   = LiteGraph.WIDGET_OUTLINE_COLOR        || "#3a3a3a";
        const textColor  = LiteGraph.WIDGET_TEXT_COLOR           || "#ddd";
        const secColor   = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR || "#999";
        const centerY    = y + H * 0.7;

        ctx.save();

        // ── background ──
        ctx.fillStyle   = bgColor;
        ctx.strokeStyle = outColor;
        ctx.lineWidth   = 1;
        ctx.beginPath();
        ctx.roundRect
            ? ctx.roundRect(x, y, bw, H, [H * 0.25])
            : ctx.rect(x, y, bw, H);
        ctx.fill();
        ctx.stroke();

        // ── content by type ──
        ctx.font = `${Math.round(H * 0.55)}px Arial`;

        if (widgetType === "button") {
            ctx.fillStyle  = textColor;
            ctx.textAlign  = "center";
            ctx.fillText(this.name, x + bw / 2, centerY);

        } else if (widgetType === "combo") {
            // left label
            ctx.fillStyle  = secColor;
            ctx.textAlign  = "left";
            ctx.fillText(this.name, x + 14, centerY);
            // right value (truncated to fit)
            ctx.fillStyle  = textColor;
            ctx.textAlign  = "right";
            const maxCh = Math.max(8, Math.floor(bw / 7.5));
            const val   = String(this.value || "");
            ctx.fillText(
                val.length > maxCh ? val.substring(0, maxCh - 1) + "…" : val,
                x + bw - 20, centerY
            );
            // arrows
            ctx.fillStyle = secColor;
            ctx.textAlign = "left";
            ctx.fillText("◂", x + 4, centerY);
            ctx.textAlign = "right";
            ctx.fillText("▸", x + bw - 4, centerY);

        } else if (widgetType === "number") {
            // left label
            ctx.fillStyle  = secColor;
            ctx.textAlign  = "left";
            ctx.fillText(this.label || this.name, x + 14, centerY);
            // right value
            ctx.fillStyle  = textColor;
            ctx.textAlign  = "right";
            ctx.fillText(Number(this.value).toFixed(3), x + bw - 20, centerY);
            // arrows
            ctx.fillStyle = secColor;
            ctx.textAlign = "left";
            ctx.fillText("◂", x + 4, centerY);
            ctx.textAlign = "right";
            ctx.fillText("▸", x + bw - 4, centerY);
        }

        ctx.restore();
    };
}

/**
 * Filter mouse events so clicks inside the image column don't
 * accidentally trigger widget interactions underneath.
 */
function addMouseGuard(widget) {
    widget.mouse = function (event, pos, node) {
        if (pos[0] < getImgColWidth(node)) return true; // consume event
        // return undefined → let LiteGraph handle normally
    };
}

// ─────────────────────────────────────────────────────────────────

app.registerExtension({
    name: "DynamicLoraLoader",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "FG_DynamicLoraLoader") return;

        const WIDGET_KEYS = [
            "spacerWidget",
            "loraWidget",
            "copyWidget",
            "modelStrWidget",
            "clipStrWidget",
            "removeWidget",
        ];

        // ── Node created ─────────────────────────────────────────

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            this.loraEntries    = [];
            this.serialize_widgets = true;
            this._previewImages = {};

            this.addLoraButton = this.addWidget("button", "➕ Add LoRA", null, () => {
                this.addLoraEntry();
            });
            // Hide from widget rendering — we draw it pinned to the
            // node floor in onDrawForeground instead.
            this.addLoraButton.computeSize = () => [0, 34];
            this.addLoraButton.draw = function () {};

            this.stackWidget = this.widgets.find((w) => w.name === "lora_stack");
            if (!this.stackWidget) {
                this.stackWidget = this.addWidget("text", "lora_stack", "[]", () => {});
                this.stackWidget.type = "converted-widget";
            }
            const stackSpacer = this.addWidget("button", "", null, () => {});
            stackSpacer.computeSize = () => [0, 10];
            stackSpacer.draw = function () {};
            this.updateStack();
        };

        // ── Helpers ──────────────────────────────────────────────

        nodeType.prototype.clearAllEntries = function () {
            while (this.loraEntries.length > 0) {
                const entry = this.loraEntries.pop();
                for (const key of WIDGET_KEYS) {
                    const idx = this.widgets.indexOf(entry[key]);
                    if (idx !== -1) this.widgets.splice(idx, 1);
                }
            }
            this.loraEntries    = [];
            this._previewImages = {};
            this.updateStack();
        };

        nodeType.prototype.updateLoraInfo = function (entry) {
            const info     = getLoraInfo(entry.loraWidget?.value);
            const triggers = info?.trigger_words || "";
            entry._triggerWords = triggers;

            if (entry.copyWidget) {
                entry.copyWidget.name = triggers
                    ? `📋 ${truncate(triggers, 42)}`
                    : "  No trigger words";
            }

            const preview = info?.preview_image || "";
            if (preview && preview.startsWith("data:image")) {
                if (
                    !this._previewImages[entry.index] ||
                    this._previewImages[entry.index].src !== preview
                ) {
                    const img = new Image();
                    img.onload = () => app.graph.setDirtyCanvas(true, true);
                    img.src    = preview;
                    this._previewImages[entry.index] = img;
                }
            } else {
                delete this._previewImages[entry.index];
            }

            app.graph.setDirtyCanvas(true, true);
        };

        // ── Add entry ────────────────────────────────────────────

        nodeType.prototype.addLoraEntry = function (config = null) {
            const index = this.loraEntries.length;
            const entry = { index, _triggerWords: "" };

            // ── spacer between entries ──
            entry.spacerWidget = this.addWidget("button", "", null, () => {});
            entry.spacerWidget.computeSize = () => [0, index === 0 ? 0 : 12];
            entry.spacerWidget.draw = function (ctx, node, w, y, H) {
                this.last_y = y;
                if (index === 0) return;
                ctx.strokeStyle = "#3a3a3a";
                ctx.lineWidth   = 1;
                ctx.beginPath();
                ctx.moveTo(10, y + H / 2);
                ctx.lineTo(w - 10, y + H / 2);
                ctx.stroke();
            };

            // ── LoRA dropdown (shifted right) ──
            entry.loraWidget = this.addWidget(
                "combo",
                `lora_${index}`,
                config?.name || loraList[0] || "None",
                () => {
                    this.updateStack();
                    this.updateLoraInfo(entry);
                },
                { values: loraList }
            );
            entry.loraWidget.draw = createShiftedDraw("combo");
            addMouseGuard(entry.loraWidget);

            // ── copy-trigger button (shifted right) ──
            entry.copyWidget = this.addWidget("button", "  No trigger words", null, () => {
                if (!entry._triggerWords) return;
                navigator.clipboard.writeText(entry._triggerWords).then(() => {
                    const prev = entry.copyWidget.name;
                    entry.copyWidget.name = "✅ Copied!";
                    app.graph.setDirtyCanvas(true, true);
                    setTimeout(() => {
                        entry.copyWidget.name = prev;
                        app.graph.setDirtyCanvas(true, true);
                    }, 1000);
                });
            });
            entry.copyWidget.draw = createShiftedDraw("button");
            addMouseGuard(entry.copyWidget);

            // ── model strength (shifted right) ──
            entry.modelStrWidget = this.addWidget(
                "number",
                `str_model_${index}`,
                config?.strength_model ?? 1.0,
                () => this.updateStack(),
                { min: -100, max: 100, step: 0.1 }
            );
            entry.modelStrWidget.draw = createShiftedDraw("number");
            addMouseGuard(entry.modelStrWidget);

            // ── clip strength (shifted right) ──
            entry.clipStrWidget = this.addWidget(
                "number",
                `str_clip_${index}`,
                config?.strength_clip ?? 1.0,
                () => this.updateStack(),
                { min: -100, max: 100, step: 0.1 }
            );
            entry.clipStrWidget.draw = createShiftedDraw("number");
            addMouseGuard(entry.clipStrWidget);

            // ── remove button (shifted right) ──
            entry.removeWidget = this.addWidget("button", "➖ Remove", null, () => {
                this.removeLoraEntry(entry);
            });
            entry.removeWidget.draw = createShiftedDraw("button");
            addMouseGuard(entry.removeWidget);

            // ── bookkeeping ──
            this.loraEntries.push(entry);
            this.updateStack();
            this.updateLoraInfo(entry);

            // keep "+ Add LoRA" at the bottom
            const addBtnIdx = this.widgets.indexOf(this.addLoraButton);
            if (addBtnIdx !== -1) {
                this.widgets.splice(addBtnIdx, 1);
                this.widgets.push(this.addLoraButton);
            }

            app.graph.setDirtyCanvas(true, true);
        };

        // ── Remove entry ─────────────────────────────────────────

        nodeType.prototype.removeLoraEntry = function (entry) {
            for (const key of WIDGET_KEYS) {
                const idx = this.widgets.indexOf(entry[key]);
                if (idx !== -1) this.widgets.splice(idx, 1);
            }

            delete this._previewImages[entry.index];
            const entryIdx = this.loraEntries.indexOf(entry);
            if (entryIdx !== -1) this.loraEntries.splice(entryIdx, 1);

            const newPreviews = {};
            this.loraEntries.forEach((e, i) => {
                if (this._previewImages[e.index]) newPreviews[i] = this._previewImages[e.index];
                e.index = i;
                if (e.loraWidget)    e.loraWidget.name    = `lora_${i}`;
                if (e.modelStrWidget) e.modelStrWidget.name = `str_model_${i}`;
                if (e.clipStrWidget)  e.clipStrWidget.name  = `str_clip_${i}`;
                // update spacer height (first entry = no gap)
                if (e.spacerWidget)  e.spacerWidget.computeSize = () => [0, i === 0 ? 0 : 12];
            });
            this._previewImages = newPreviews;

            this.updateStack();
            app.graph.setDirtyCanvas(true, true);
        };

        // ── Stack serialisation ──────────────────────────────────

        nodeType.prototype.updateStack = function () {
            const stack = this.loraEntries.map((e) => ({
                name:           e.loraWidget?.value       || "",
                strength_model: e.modelStrWidget?.value   ?? 1.0,
                strength_clip:  e.clipStrWidget?.value    ?? 1.0,
            }));
            if (this.stackWidget) {
                this.stackWidget.value = JSON.stringify(stack);
            }
        };

        // ── Left-column preview image ────────────────────────────
        //  Drawn in onDrawForeground so it paints over the now-
        //  empty left column area (entry widgets render right of it).

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);

            // ── Entry preview images ──
            if (this.loraEntries?.length) {

            const imgW = getImgColWidth(this);
            const PAD   = 5;

            for (const entry of this.loraEntries) {
                const topW = entry.loraWidget;
                const botW = entry.removeWidget;
                if (!topW || !botW) continue;
                if (topW.last_y === undefined || botW.last_y === undefined) continue;

                const widgetH = LiteGraph.NODE_WIDGET_HEIGHT || 20;
                const top     = topW.last_y - 3;
                const bottom  = botW.last_y + widgetH + 6;

                const rx = PAD;
                const ry = top;
                const rw = imgW - PAD * 2;
                const rh = bottom - top;

                // ── dark panel background ──
                ctx.fillStyle = "#1a1a2e";
                ctx.beginPath();
                ctx.roundRect
                    ? ctx.roundRect(rx, ry, rw, rh, 6)
                    : ctx.rect(rx, ry, rw, rh);
                ctx.fill();

                // ── image or fallback ──
                const img = this._previewImages?.[entry.index];
                if (img && img.complete && img.naturalWidth) {
                    const inW  = rw - 4;
                    const inH  = rh - 4;
                    const scale = Math.min(inW / img.naturalWidth, inH / img.naturalHeight);
                    const dw   = img.naturalWidth  * scale;
                    const dh   = img.naturalHeight * scale;
                    const dx   = rx + (rw - dw) / 2;
                    const dy   = ry + (rh - dh) / 2;

                    ctx.save();
                    ctx.beginPath();
                    ctx.roundRect
                        ? ctx.roundRect(rx + 2, ry + 2, rw - 4, rh - 4, 4)
                        : ctx.rect(rx + 2, ry + 2, rw - 4, rh - 4);
                    ctx.clip();
                    ctx.drawImage(img, dx, dy, dw, dh);
                    ctx.restore();
                } else {
                    ctx.save();
                    ctx.fillStyle    = "#555";
                    ctx.font         = "bold 11px Arial";
                    ctx.textAlign    = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText("NO",      rx + rw / 2, ry + rh / 2 - 8);
                    ctx.fillText("PREVIEW", rx + rw / 2, ry + rh / 2 + 8);
                    ctx.restore();
                }

                // ── border ──
                ctx.strokeStyle = "#444";
                ctx.lineWidth   = 1;
                ctx.beginPath();
                ctx.roundRect
                    ? ctx.roundRect(rx, ry, rw, rh, 6)
                    : ctx.rect(rx, ry, rw, rh);
                ctx.stroke();
            }

            } // end if (entries)

            // ── "+ Add LoRA" button pinned to node floor ──
            const btnH      = 26;
            const btnPad    = 8;
            const btnMargin = 15;
            const btnY      = this.size[1] - btnH - btnPad;
            const btnX      = btnMargin;
            const btnW      = this.size[0] - btnMargin * 2;

            ctx.fillStyle   = LiteGraph.WIDGET_BGCOLOR       || "#232323";
            ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR  || "#3a3a3a";
            ctx.lineWidth   = 1;
            ctx.beginPath();
            ctx.roundRect
                ? ctx.roundRect(btnX, btnY, btnW, btnH, [btnH * 0.25])
                : ctx.rect(btnX, btnY, btnW, btnH);
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || "#ddd";
            ctx.font      = `${Math.round(btnH * 0.55)}px Arial`;
            ctx.textAlign = "center";
            ctx.fillText("➕ Add LoRA", btnX + btnW / 2, btnY + btnH * 0.7);

            // Store hit zone for onMouseDown
            this._addBtnHitZone = { x: btnX, y: btnY, w: btnW, h: btnH };
        };

        // ── Click handler for the floor-pinned Add button ─────────

        const onMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (e, pos, graphCanvas) {
            const z = this._addBtnHitZone;
            if (z && pos[0] >= z.x && pos[0] <= z.x + z.w &&
                     pos[1] >= z.y && pos[1] <= z.y + z.h) {
                this.addLoraEntry();
                return true;
            }
            if (onMouseDown) return onMouseDown.apply(this, arguments);
        };

        // ── Workflow restore ─────────────────────────────────────

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            onConfigure?.apply(this, arguments);
            this.clearAllEntries();

            if (info.widgets_values) {
                const stackVal = info.widgets_values.find(
                    (v) => typeof v === "string" && v.startsWith("[")
                );
                if (stackVal) {
                    try {
                        JSON.parse(stackVal).forEach((cfg) => this.addLoraEntry(cfg));
                    } catch (e) {}
                }
            }
        };
    },
});