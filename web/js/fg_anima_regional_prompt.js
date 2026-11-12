const { app } = window.comfyAPI.app;
import { getDrawColor } from "./utils.js";

// Visual builder for FG_AnimaRegionalPrompt. The json_prompt widget stays the
// single source of truth: the builder parses it when shown and writes it back
// on every edit, so the Python side is unchanged.

const NODE_CLASS = "FG_AnimaRegionalPrompt";
const SNAP = 16;            // Anima attends in 16px patches
const GRID = 64;
const HANDLE = 8;           // corner grab distance, screen px
const MAX_CANVAS_H = 420;   // css px

const STYLE = `
.fg-arp { display:flex; flex-direction:column; gap:4px; font:12px sans-serif; color:var(--input-text, #ddd); }
.fg-arp textarea, .fg-arp input, .fg-arp select, .fg-arp button {
    background:var(--comfy-input-bg, #222); color:inherit; border:1px solid var(--border-color, #444);
    border-radius:4px; padding:2px 4px; font:inherit; box-sizing:border-box; }
.fg-arp textarea { width:100%; resize:vertical; min-height:38px; }
.fg-arp button { padding:2px 8px; cursor:pointer; }
.fg-arp button:disabled { opacity:0.4; cursor:default; }
.fg-arp .row { display:flex; gap:4px; align-items:center; }
.fg-arp .row > label { display:flex; align-items:center; gap:3px; flex:1 1 0; min-width:0; }
.fg-arp .row > input, .fg-arp .row > select { flex:2 1 0; min-width:0; }
.fg-arp .row > label > input { flex:1 1 0; width:0; min-width:0; }
.fg-arp canvas { display:block; margin:0 auto; border-radius:4px; touch-action:none; cursor:crosshair; }
.fg-arp .hint { opacity:0.6; font-size:11px; text-align:center; }
.fg-arp .err { color:#e66; white-space:pre-wrap; }
`;

function el(tag, props = {}, ...children) {
    const e = Object.assign(document.createElement(tag), props);
    e.append(...children);
    return e;
}

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const snap = (v) => Math.round(v / SNAP) * SNAP;

function setupBuilder(node) {
    const jsonWidget = node.widgets.find((w) => w.name === "json_prompt");
    const toggle = node.widgets.find((w) => w.name === "use_builder");
    if (!jsonWidget || !toggle || node.fgRegionBuilder) return;

    if (!document.getElementById("fg-arp-style")) {
        document.head.append(el("style", { id: "fg-arp-style", textContent: STYLE }));
    }

    let data = null;
    let selected = 0;
    let drag = null;

    const canvas = el("canvas");
    const error = el("div", { className: "err" });
    const width = el("input", { type: "number", min: SNAP, step: SNAP });
    const height = el("input", { type: "number", min: SNAP, step: SNAP });
    const base = el("textarea", { placeholder: "Base prompt" });
    const negative = el("textarea", { placeholder: "Negative prompt" });
    const regionSelect = el("select");
    const addBtn = el("button", { textContent: "+ Add" });
    const removeBtn = el("button", { textContent: "✕ Remove" });
    const name = el("input", { type: "text", placeholder: "Name" });
    const strength = el("input", { type: "number", min: 0, max: 10, step: 0.05 });
    const rx = el("input", { type: "number", min: 0, step: SNAP });
    const ry = el("input", { type: "number", min: 0, step: SNAP });
    const rw = el("input", { type: "number", min: 0, step: SNAP });
    const rh = el("input", { type: "number", min: 0, step: SNAP });
    const prompt = el("textarea", { placeholder: "Region prompt" });
    const regionFields = [name, strength, rx, ry, rw, rh, prompt, removeBtn];

    const body = el("div", { className: "fg-arp" },
        el("div", { className: "row" }, el("label", {}, "W", width), el("label", {}, "H", height)),
        base,
        negative,
        canvas,
        el("div", { className: "hint", textContent: "Drag empty space to draw a region · drag a region to move · drag a corner to resize" }),
        el("div", { className: "row" }, regionSelect, addBtn, removeBtn),
        el("div", { className: "row" }, name, el("label", {}, "Str", strength)),
        el("div", { className: "row" },
            el("label", {}, "X", rx), el("label", {}, "Y", ry),
            el("label", {}, "W", rw), el("label", {}, "H", rh)),
        prompt,
    );
    const inner = el("div", {}, error, body);
    const root = el("div", {}, inner);

    const panel = node.addDOMWidget("region_builder", "fg_region_builder", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => inner.offsetHeight + 4,
    });
    panel.serialize = false;

    // ── JSON <-> builder ────────────────────────────────────────────

    function load() {
        try {
            const parsed = JSON.parse(jsonWidget.value);
            if (!Array.isArray(parsed.regions)) throw new Error("\"regions\" must be a list.");
            data = parsed;
            data.width = Math.round(Number(data.width)) || 1024;
            data.height = Math.round(Number(data.height)) || 1024;
            data.base_prompt ??= "";
            data.negative_prompt ??= "";
            error.textContent = "";
            body.style.display = "";
        } catch (e) {
            data = null;
            error.textContent = `json_prompt is not valid: ${e.message}\nTurn the builder off to fix it by hand.`;
            body.style.display = "none";
        }
        selected = data ? clamp(selected, 0, Math.max(0, data.regions.length - 1)) : 0;
        fillForm();
        requestAnimationFrame(draw);
    }

    function save() {
        jsonWidget.value = JSON.stringify(data, null, 2);
        node.setDirtyCanvas(true, true);
    }

    function fillForm() {
        if (!data) return;
        width.value = data.width;
        height.value = data.height;
        base.value = data.base_prompt;
        negative.value = data.negative_prompt;
        regionSelect.replaceChildren(...data.regions.map((r, i) =>
            el("option", { value: i, textContent: `${i + 1}. ${r.name || "Region"}` })));
        regionSelect.value = selected;
        fillRegion();
    }

    function fillRegion() {
        const r = data?.regions[selected];
        for (const f of regionFields) f.disabled = !r;
        name.value = r?.name ?? "";
        strength.value = r?.strength ?? "";
        rx.value = r?.x ?? "";
        ry.value = r?.y ?? "";
        rw.value = r?.width ?? "";
        rh.value = r?.height ?? "";
        prompt.value = r?.prompt ?? "";
    }

    // ── Canvas ──────────────────────────────────────────────────────

    function draw() {
        if (!data || root.style.display === "none") return;
        const aspect = data.width / data.height;
        const cssW = Math.max(1, Math.min(body.clientWidth, MAX_CANVAS_H * aspect));
        const cssH = cssW / aspect;
        const dpr = window.devicePixelRatio || 1;
        canvas.style.width = `${cssW}px`;
        canvas.style.height = `${cssH}px`;
        canvas.width = Math.round(cssW * dpr);
        canvas.height = Math.round(cssH * dpr);

        const s = cssW / data.width;
        const ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        ctx.fillStyle = "#1b1b1b";
        ctx.fillRect(0, 0, cssW, cssH);
        ctx.beginPath();
        for (let x = GRID; x < data.width; x += GRID) { ctx.moveTo(x * s, 0); ctx.lineTo(x * s, cssH); }
        for (let y = GRID; y < data.height; y += GRID) { ctx.moveTo(0, y * s); ctx.lineTo(cssW, y * s); }
        ctx.strokeStyle = "#ffffff14";
        ctx.lineWidth = 1;
        ctx.stroke();

        const count = data.regions.length;
        const order = [...data.regions.keys()].filter((i) => i !== selected);
        if (selected < count) order.push(selected);
        ctx.font = "11px sans-serif";
        ctx.textBaseline = "top";
        for (const i of order) {
            const r = data.regions[i];
            const [x, y, w, h] = [r.x * s, r.y * s, r.width * s, r.height * s];
            const isSel = i === selected;
            ctx.fillStyle = getDrawColor(i / count, isSel ? "90" : "50");
            ctx.fillRect(x, y, w, h);
            ctx.strokeStyle = isSel ? "#ffffff" : getDrawColor(i / count, "FF");
            ctx.lineWidth = isSel ? 2 : 1;
            ctx.strokeRect(x, y, w, h);
            ctx.fillStyle = "#ffffff";
            ctx.fillText(`${i + 1}. ${r.name || ""}`, x + 4, y + 4, Math.max(0, w - 8));
            if (isSel) {
                for (const [cx, cy] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) {
                    ctx.fillRect(cx - 3, cy - 3, 6, 6);
                }
            }
        }
    }

    function toImage(e) {
        const rect = canvas.getBoundingClientRect();
        return [
            (e.clientX - rect.left) * data.width / rect.width,
            (e.clientY - rect.top) * data.height / rect.height,
        ];
    }

    // Opposite corner of the corner under p, i.e. the anchor for a resize.
    function cornerAnchor(r, [px, py], tol) {
        for (const [cx, ax] of [[r.x, r.x + r.width], [r.x + r.width, r.x]]) {
            for (const [cy, ay] of [[r.y, r.y + r.height], [r.y + r.height, r.y]]) {
                if (Math.abs(px - cx) <= tol && Math.abs(py - cy) <= tol) return [ax, ay];
            }
        }
        return null;
    }

    function regionAt([px, py]) {
        const inside = (r) => px >= r.x && px <= r.x + r.width && py >= r.y && py <= r.y + r.height;
        if (data.regions[selected] && inside(data.regions[selected])) return selected;
        for (let i = data.regions.length - 1; i >= 0; i--) {
            if (inside(data.regions[i])) return i;
        }
        return -1;
    }

    function tolerance() {
        return HANDLE * data.width / canvas.getBoundingClientRect().width;
    }

    function newRegion(x, y, w, h) {
        return { name: `Region ${data.regions.length + 1}`, prompt: "", x, y, width: w, height: h, strength: 0.85 };
    }

    canvas.addEventListener("pointerdown", (e) => {
        if (!data || e.button !== 0) return;
        e.preventDefault();
        e.stopPropagation();
        const p = toImage(e);
        const anchor = data.regions[selected] && cornerAnchor(data.regions[selected], p, tolerance());
        if (anchor) {
            drag = { anchor };
        } else {
            const hit = regionAt(p);
            if (hit !== -1) {
                selected = hit;
                const r = data.regions[hit];
                drag = { offset: [p[0] - r.x, p[1] - r.y] };
            } else {
                const start = [clamp(snap(p[0]), 0, data.width), clamp(snap(p[1]), 0, data.height)];
                data.regions.push(newRegion(start[0], start[1], 0, 0));
                selected = data.regions.length - 1;
                drag = { anchor: start, created: true };
            }
        }
        canvas.setPointerCapture(e.pointerId);
        fillForm();
        draw();
    });

    canvas.addEventListener("pointermove", (e) => {
        if (!data) return;
        const p = toImage(e);
        if (!drag) {
            const r = data.regions[selected];
            const anchor = r && cornerAnchor(r, p, tolerance());
            canvas.style.cursor = anchor
                ? ((anchor[0] < p[0]) === (anchor[1] < p[1]) ? "nwse-resize" : "nesw-resize")
                : regionAt(p) !== -1 ? "move" : "crosshair";
            return;
        }
        const r = data.regions[selected];
        if (drag.anchor) {
            const qx = clamp(snap(p[0]), 0, data.width);
            const qy = clamp(snap(p[1]), 0, data.height);
            r.x = Math.min(drag.anchor[0], qx);
            r.y = Math.min(drag.anchor[1], qy);
            r.width = Math.abs(qx - drag.anchor[0]);
            r.height = Math.abs(qy - drag.anchor[1]);
        } else {
            r.x = clamp(snap(p[0] - drag.offset[0]), 0, Math.max(0, data.width - r.width));
            r.y = clamp(snap(p[1] - drag.offset[1]), 0, Math.max(0, data.height - r.height));
        }
        fillRegion();
        draw();
    });

    function endDrag(e) {
        if (!drag) return;
        const r = data.regions[selected];
        // A click on empty space draws nothing, so drop the empty region it started.
        if (drag.created && (r.width < SNAP || r.height < SNAP)) {
            data.regions.pop();
            selected = Math.max(0, data.regions.length - 1);
        }
        drag = null;
        canvas.releasePointerCapture?.(e.pointerId);
        save();
        fillForm();
        draw();
    }
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    // ── Form ────────────────────────────────────────────────────────

    function onCanvasSize(key, input) {
        input.addEventListener("change", () => {
            const old = data[key];
            const v = Math.max(SNAP, Math.round(Number(input.value)) || old);
            const f = v / old;
            for (const r of data.regions) {
                if (key === "width") { r.x = Math.round(r.x * f); r.width = Math.round(r.width * f); }
                else { r.y = Math.round(r.y * f); r.height = Math.round(r.height * f); }
            }
            data[key] = v;
            save();
            fillForm();
            draw();
        });
    }
    onCanvasSize("width", width);
    onCanvasSize("height", height);

    base.addEventListener("input", () => { data.base_prompt = base.value; save(); });
    negative.addEventListener("input", () => { data.negative_prompt = negative.value; save(); });
    prompt.addEventListener("input", () => { data.regions[selected].prompt = prompt.value; save(); });
    name.addEventListener("input", () => {
        data.regions[selected].name = name.value;
        regionSelect.options[selected].textContent = `${selected + 1}. ${name.value || "Region"}`;
        save();
        draw();
    });
    strength.addEventListener("change", () => {
        const r = data.regions[selected];
        r.strength = Math.max(0, Number(strength.value) || 0);
        save();
        fillRegion();
    });
    for (const [key, input] of [["x", rx], ["y", ry], ["width", rw], ["height", rh]]) {
        input.addEventListener("change", () => {
            data.regions[selected][key] = Math.max(0, Math.round(Number(input.value)) || 0);
            save();
            fillRegion();
            draw();
        });
    }

    regionSelect.addEventListener("change", () => {
        selected = Number(regionSelect.value);
        fillRegion();
        draw();
    });
    addBtn.addEventListener("click", () => {
        const w = snap(data.width / 2), h = snap(data.height / 2);
        data.regions.push(newRegion(snap((data.width - w) / 2), snap((data.height - h) / 2), w, h));
        selected = data.regions.length - 1;
        save();
        fillForm();
        draw();
    });
    removeBtn.addEventListener("click", () => {
        data.regions.splice(selected, 1);
        selected = Math.max(0, selected - 1);
        save();
        fillForm();
        draw();
    });

    // ── Toggle + sizing ─────────────────────────────────────────────

    function applyMode() {
        const on = !!toggle.value;
        jsonWidget.hidden = on;
        if (jsonWidget.element) jsonWidget.element.style.display = on ? "none" : "";
        panel.hidden = !on;
        root.style.display = on ? "" : "none";
        if (on) load();
        requestAnimationFrame(() => {
            node.setSize([node.size[0], node.computeSize()[1]]);
            node.setDirtyCanvas(true, true);
        });
    }

    let lastWidth = 0;
    new ResizeObserver(() => {
        if (body.clientWidth !== lastWidth) {
            lastWidth = body.clientWidth;
            draw();
        }
        const need = node.computeSize()[1];
        if (panel.hidden !== true && node.size[1] < need) node.setSize([node.size[0], need]);
    }).observe(inner);

    const toggleCallback = toggle.callback;
    toggle.callback = function () {
        const r = toggleCallback?.apply(this, arguments);
        applyMode();
        return r;
    };

    node.fgRegionBuilder = { applyMode };
    applyMode();
}

app.registerExtension({
    name: "FG.AnimaRegionalPrompt.Builder",

    async nodeCreated(node) {
        if (node.comfyClass === NODE_CLASS) setupBuilder(node);
    },

    // Saved widget values land after nodeCreated, so re-read the toggle and JSON.
    async loadedGraphNode(node) {
        if (node.comfyClass === NODE_CLASS) node.fgRegionBuilder?.applyMode();
    },
});
