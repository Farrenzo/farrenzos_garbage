const { api } = window.comfyAPI.api;
const { app } = window.comfyAPI.app;

// Widget name -> model folder type, for any node that picks a model by name.
const MODEL_WIDGETS = {
    ckpt_name:        "checkpoints",
    unet_name:        "diffusion_models",
    lora_name:        "loras",
    vae_name:         "vae",
    clip_name:        "text_encoders",
    clip_name1:       "text_encoders",
    clip_name2:       "text_encoders",
    clip_name3:       "text_encoders",
    clip_name4:       "text_encoders",
    control_net_name: "controlnet",
};
// Nodes with one of these get the preview + recommended settings panel.
const PANEL_WIDGETS = ["ckpt_name", "unet_name"];
const SETTING_KEYS = ["sampler", "scheduler", "steps", "cfg", "clip_skip", "resolution", "shift", "denoise", "strength_model", "strength_clip"];

function widgetKind(name) {
    // FG_DynamicLoraLoader names its entries lora_0, lora_1, ...
    return /^lora_\d+$/.test(name) ? "loras" : MODEL_WIDGETS[name];
}

// ── Library cache ────────────────────────────────────────────────

const library = {};   // kind -> { "folder\\name.safetensors": info }
const loading = {};   // kind -> Promise

async function loadLibrary(kind) {
    try {
        const resp = await api.fetchApi(`/fg/library/kind/${kind}`);
        library[kind] = resp.ok ? await resp.json() : {};
    } catch (e) {
        console.error(`Failed to load resource library for ${kind}:`, e);
        library[kind] = {};
    }
}

export function ensureLibrary(kind) {
    return (loading[kind] ??= loadLibrary(kind));
}

export function findModel(kind, name) {
    const entries = library[kind];
    if (!entries || !name) return null;
    return entries[name] ?? entries[name.replace(/\//g, "\\")] ?? entries[name.replace(/\\/g, "/")] ?? null;
}

async function refreshLibrary(kind) {
    loading[kind] = loadLibrary(kind);
    await loading[kind];
    for (const node of app.graph._nodes) node.onLibraryUpdated?.();
}

// ── Edit info dialog ─────────────────────────────────────────────

function el(tag, props = {}, ...children) {
    const e = Object.assign(document.createElement(tag), props);
    e.append(...children);
    return e;
}

document.head.append(el("style", { textContent: `
.fg-rm-overlay { position: fixed; inset: 0; z-index: 10000; background: rgba(0,0,0,.55); display: flex; align-items: center; justify-content: center; }
.fg-rm-dialog { width: min(760px, 94vw); max-height: 92vh; overflow: auto; background: var(--comfy-menu-bg); color: var(--fg-color); border: 1px solid var(--border-color); border-radius: 8px; padding: 16px; font: 13px sans-serif; box-shadow: 0 8px 32px rgba(0,0,0,.5); }
.fg-rm-head { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
.fg-rm-title { font-size: 16px; font-weight: bold; word-break: break-all; }
.fg-rm-sub { color: var(--descrip-text, #999); font-size: 11px; margin-top: 2px; }
.fg-rm-body { display: grid; grid-template-columns: 220px 1fr; gap: 14px; }
.fg-rm-preview { height: 280px; border: 2px dashed var(--border-color); border-radius: 6px; display: flex; align-items: center; justify-content: center; text-align: center; cursor: pointer; overflow: hidden; color: var(--descrip-text, #999); padding: 6px; white-space: pre-line; }
.fg-rm-preview.fg-rm-drag { border-color: var(--fg-color); }
.fg-rm-preview img { max-width: 100%; max-height: 100%; object-fit: contain; }
.fg-rm-fields label { display: block; color: var(--descrip-text, #999); font-size: 11px; margin: 8px 0 3px; }
.fg-rm-fields label:first-child { margin-top: 0; }
.fg-rm-dialog input, .fg-rm-dialog textarea { width: 100%; box-sizing: border-box; background: var(--comfy-input-bg); color: var(--input-text); border: 1px solid var(--border-color); border-radius: 4px; padding: 5px 7px; font: inherit; }
.fg-rm-dialog textarea { resize: vertical; }
.fg-rm-settings { margin-top: 14px; }
.fg-rm-row { display: grid; grid-template-columns: 180px 1fr 28px; gap: 6px; margin-bottom: 6px; }
.fg-rm-dialog button { background: var(--comfy-input-bg); color: var(--input-text); border: 1px solid var(--border-color); border-radius: 4px; padding: 5px 12px; cursor: pointer; }
.fg-rm-dialog button:hover { filter: brightness(1.3); }
.fg-rm-foot { display: flex; justify-content: flex-end; align-items: center; gap: 8px; margin-top: 16px; }
.fg-rm-error { color: #e66; margin-right: auto; }
.fg-rm-panel { display: flex; gap: 8px; height: 100%; box-sizing: border-box; padding: 2px 4px; font: 12px sans-serif; color: var(--fg-color); overflow: hidden; }
.fg-rm-panel img { max-width: 45%; height: 100%; object-fit: contain; border-radius: 4px; }
.fg-rm-panel-text { overflow: auto; line-height: 1.5; }
.fg-rm-panel-text b { color: var(--descrip-text, #999); font-weight: normal; }
@media (max-width: 560px) { .fg-rm-body { grid-template-columns: 1fr; } .fg-rm-row { grid-template-columns: 1fr 1fr 28px; } }
` }));

export async function openEditInfo(kind, name) {
    await ensureLibrary(kind);
    const info = findModel(kind, name);
    const close = () => overlay.remove();
    const dialog = el("div", { className: "fg-rm-dialog" });
    const overlay = el("div", { className: "fg-rm-overlay", tabIndex: -1 }, dialog);
    overlay.addEventListener("mousedown", (e) => e.target === overlay && close());
    overlay.addEventListener("keydown", (e) => e.key === "Escape" && close());

    const fileName = name.replace(/\\/g, "/").split("/").pop();
    dialog.append(el("div", { className: "fg-rm-head" },
        el("div", {},
            el("div", { className: "fg-rm-title", textContent: fileName }),
            el("div", { className: "fg-rm-sub", textContent: info ? `${kind} · sha256 ${info.sha256.slice(0, 12)}…` : kind, title: info?.sha256 ?? "" }),
        ),
        el("button", { textContent: "✕", onclick: close }),
    ));

    if (!info) {
        dialog.append(
            el("div", { textContent: "This model isn't in the resource library yet. Run the 🗑️ Resource Manager node to scan it, then try again." }),
            el("div", { className: "fg-rm-foot" }, el("button", { textContent: "Close", onclick: close })),
        );
        document.body.append(overlay);
        overlay.focus();
        return;
    }

    // ── preview: click, drop or paste ──
    let pendingImage = null;
    const preview = el("div", { className: "fg-rm-preview", title: "Click, drop or paste an image" });
    const showPreview = (src) => preview.replaceChildren(src ? el("img", { src }) : "No preview\n\nClick, drop or paste an image");
    showPreview(info.preview_url);
    const picker = el("input", { type: "file", accept: "image/*", hidden: true });
    const takeImage = (file) => {
        if (!file?.type.startsWith("image/")) return;
        pendingImage = file;
        showPreview(URL.createObjectURL(file));
    };
    picker.onchange = () => takeImage(picker.files[0]);
    preview.onclick = () => picker.click();
    preview.ondragover = (e) => { e.preventDefault(); preview.classList.add("fg-rm-drag"); };
    preview.ondragleave = () => preview.classList.remove("fg-rm-drag");
    preview.ondrop = (e) => { e.preventDefault(); preview.classList.remove("fg-rm-drag"); takeImage(e.dataTransfer.files[0]); };
    // Keep pastes and drops away from ComfyUI's document handlers (Load Image node / workflow loading).
    overlay.addEventListener("paste", (e) => {
        e.stopPropagation();
        const file = [...e.clipboardData.files].find((f) => f.type.startsWith("image/"));
        if (file) { e.preventDefault(); takeImage(file); }
    });
    overlay.addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); });
    overlay.addEventListener("drop", (e) => { e.preventDefault(); e.stopPropagation(); });

    // ── text fields ──
    const baseModel = el("input", { value: info.base_model, placeholder: "e.g. Illustrious, Anima, Flux" });
    const sourceUrl = el("input", { value: info.source_url, placeholder: "https://…" });
    const triggers  = el("textarea", { value: info.trigger_words, rows: 3 });
    const notes     = el("textarea", { value: info.notes, rows: 4 });

    // ── recommended settings ──
    const keyList = el("datalist", { id: "fg-rm-setting-keys" }, ...SETTING_KEYS.map((k) => el("option", { value: k })));
    const rows = el("div");
    const addRow = (key = "", value = "") => {
        const row = el("div", { className: "fg-rm-row" },
            el("input", { value: key, placeholder: "setting" }),
            el("input", { value: String(value), placeholder: "value" }),
            el("button", { textContent: "✕", title: "Remove", onclick: () => row.remove() }),
        );
        row.firstChild.setAttribute("list", keyList.id);
        rows.append(row);
    };
    Object.entries(info.settings).forEach(([k, v]) => addRow(k, v));

    const error = el("span", { className: "fg-rm-error" });
    const save = async () => {
        const settings = {};
        for (const row of rows.children) {
            const [k, v] = [...row.querySelectorAll("input")].map((i) => i.value.trim());
            if (k) settings[k] = v;
        }
        let resp = await api.fetchApi(`/fg/library/model/${info.sha256}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ base_model: baseModel.value.trim(), source_url: sourceUrl.value.trim(), trigger_words: triggers.value.trim(), notes: notes.value.trim(), settings }),
        });
        if (resp.ok && pendingImage) {
            const form = new FormData();
            form.append("image", pendingImage, pendingImage.name || "preview.png");
            resp = await api.fetchApi(`/fg/library/model/${info.sha256}/preview`, { method: "POST", body: form });
        }
        if (!resp.ok) {
            error.textContent = (await resp.json().catch(() => null))?.error ?? `Save failed (${resp.status})`;
            return;
        }
        close();
        await refreshLibrary(kind);
    };

    dialog.append(
        el("div", { className: "fg-rm-body" },
            el("div", {}, preview, picker),
            el("div", { className: "fg-rm-fields" },
                el("label", { textContent: "Base model" }), baseModel,
                el("label", { textContent: "Source URL" }), sourceUrl,
                el("label", { textContent: "Trigger words" }), triggers,
                el("label", { textContent: "Notes" }), notes,
            ),
        ),
        el("div", { className: "fg-rm-settings" },
            el("label", { textContent: "Recommended settings", style: "display:block;color:var(--descrip-text,#999);font-size:11px;margin-bottom:6px" }),
            keyList, rows,
            el("button", { textContent: "➕ Add setting", onclick: () => addRow() }),
        ),
        el("div", { className: "fg-rm-foot" }, error,
            el("button", { textContent: "Cancel", onclick: close }),
            el("button", { textContent: "💾 Save", onclick: save }),
        ),
    );
    document.body.append(overlay);
    overlay.focus();
}

// ── Preview + recommended settings panel ─────────────────────────

function addInfoPanel(node, widget, kind) {
    const text = el("div", { className: "fg-rm-panel-text" });
    const img = el("img");
    const root = el("div", { className: "fg-rm-panel" }, img, text);
    let hasInfo = false;
    const panel = node.addDOMWidget("fg_model_info", "fg_model_info", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => (hasInfo ? 140 : 22),
    });
    panel.serialize = false;

    const refresh = () => {
        const info = findModel(kind, widget.value);
        const settings = Object.entries(info?.settings ?? {});
        hasInfo = Boolean(info?.preview_url || info?.base_model || settings.length);
        img.style.display = info?.preview_url ? "" : "none";
        if (info?.preview_url) img.src = info.preview_url;
        text.replaceChildren(...(hasInfo
            ? [info.base_model && ["base", info.base_model], ...settings].filter(Boolean)
                .map(([k, v]) => el("div", {}, el("b", { textContent: `${k}: ` }), String(v)))
            : [el("b", { textContent: "No info · right-click › 📝 Edit info" })]));
        node.setSize([node.size[0], Math.max(node.size[1], node.computeSize()[1])]);
        node.setDirtyCanvas(true, true);
    };
    const callback = widget.callback;
    widget.callback = function () {
        const r = callback?.apply(this, arguments);
        refresh();
        return r;
    };
    return refresh;
}

// ── Registration ─────────────────────────────────────────────────

app.registerExtension({
    name: "FG.ResourceManager",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const inputs = Object.keys({ ...nodeData.input?.required, ...nodeData.input?.optional });
        if (!inputs.some(widgetKind) && nodeData.name !== "FG_DynamicLoraLoader") return;

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            const r = getExtraMenuOptions?.apply(this, arguments);
            const targets = (this.widgets ?? []).filter((w) => widgetKind(w.name) && typeof w.value === "string" && w.value && w.value !== "None");
            const edit = (w) => () => openEditInfo(widgetKind(w.name), w.value);
            if (targets.length === 1) {
                options.push({ content: "📝 Edit info", callback: edit(targets[0]) });
            } else if (targets.length > 1) {
                options.push({
                    content: "📝 Edit info",
                    has_submenu: true,
                    submenu: { options: targets.map((w) => ({ content: w.value.replace(/\\/g, "/").split("/").pop(), callback: edit(w) })) },
                });
            }
            return r;
        };

        const panelWidget = PANEL_WIDGETS.find((n) => inputs.includes(n));
        if (!panelWidget) return;
        const kind = MODEL_WIDGETS[panelWidget];
        await ensureLibrary(kind);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const widget = this.widgets?.find((w) => w.name === panelWidget);
            if (widget) {
                this.onLibraryUpdated = addInfoPanel(this, widget, kind);
                this.onLibraryUpdated();
            }
            return r;
        };

        // Restored workflows set widget values without firing callbacks.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            this.onLibraryUpdated?.();
            return r;
        };
    },
});
