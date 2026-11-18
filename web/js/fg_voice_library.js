const { api } = window.comfyAPI.api;
const { app } = window.comfyAPI.app;

// 🗑️ Audio Library Loader (FG_VoiceLibrary): the folder's voices as a checkbox list.
// The scan is cached server-side (nodes/fg_audio_nodes.py); ⟳ re-scans the folder.
// Ticked voices live in the hidden "selected_voices" widget as a JSON list of ids; ticked voices
// become speakers 0, 1, ... in list order (a JSON dialogue picks its own voices and ticks them).

const NODE_CLASS = "FG_VoiceLibrary";

function el(tag, props = {}, ...children) {
    const e = Object.assign(document.createElement(tag), props);
    e.append(...children);
    return e;
}

document.head.append(el("style", { textContent: `
.fg-vl { font: 12px sans-serif; color: var(--fg-color); display: flex; flex-direction: column; gap: 4px; padding: 2px 4px; }
.fg-vl-bar { display: flex; gap: 8px; align-items: center; color: var(--descrip-text, #999); font-size: 11px; }
.fg-vl-bar a { color: var(--descrip-text, #999); cursor: pointer; text-decoration: underline; }
.fg-vl-bar .grow { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.fg-vl-list { max-height: 260px; overflow-y: auto; border: 1px solid var(--border-color); border-radius: 4px; background: var(--comfy-input-bg); }
.fg-vl-group { padding: 4px 6px 2px; font-size: 10px; text-transform: uppercase; letter-spacing: .08em; color: var(--descrip-text, #999); }
.fg-vl-row { display: grid; grid-template-columns: 16px 22px minmax(0, 1fr) auto; gap: 6px; align-items: center; padding: 2px 6px; cursor: pointer; }
.fg-vl-row:hover { background: rgba(127,127,127,.12); }
.fg-vl-row input { margin: 0; }
.fg-vl-row .spk { font-size: 10px; color: var(--descrip-text, #999); text-align: right; }
.fg-vl-row .files { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.fg-vl-row .dur { font-size: 10px; color: var(--descrip-text, #999); }
.fg-vl-empty { padding: 8px; color: var(--descrip-text, #999); }
.fg-vl-error { color: #e57373; }
` }));

function setupLibrary(node) {
    const folderW = node.widgets.find((w) => w.name === "folder_path");
    const selW = node.widgets.find((w) => w.name === "selected_voices");
    if (!folderW || !selW) return;
    selW.hidden = true;
    selW.type = "converted-widget";
    selW.computeSize = () => [0, -4];
    if (selW.element) selW.element.style.display = "none";

    let voices = [];
    const status = el("span", { className: "grow" });
    const all = el("a", { textContent: "all" });
    const none = el("a", { textContent: "none" });
    const list = el("div", { className: "fg-vl-list" });
    const root = el("div", { className: "fg-vl" }, el("div", { className: "fg-vl-bar" }, status, all, none), list);
    const panel = node.addDOMWidget("voice_list", "fg_voice_list", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => Math.min(300, 40 + voices.length * 21),
    });
    panel.serialize = false;

    const selected = () => {
        try {
            const v = JSON.parse(selW.value || "[]");
            return Array.isArray(v) ? v.map(String) : [];
        } catch {
            return [];
        }
    };
    const setSelected = (ids) => {
        selW.value = JSON.stringify(voices.map((v) => v.id).filter((id) => ids.includes(id)));
        render();
        node.setDirtyCanvas(true, true);
    };

    function render() {
        const ticked = selected();
        list.replaceChildren();
        if (!voices.length) {
            list.append(el("div", { className: "fg-vl-empty", textContent: folderW.value ? "No voices: each needs audio + its .txt (a.wav + a.wav.txt)" : "Set folder_path" }));
            return;
        }
        let group = null;
        let speaker = 0;
        for (const v of voices) {
            const g = v.id.includes("/") ? v.id.slice(0, v.id.lastIndexOf("/")) : "";
            if (g !== group) {
                group = g;
                if (g || voices.some((x) => x.id.includes("/"))) list.append(el("div", { className: "fg-vl-group", textContent: g || "(top level)" }));
            }
            const on = ticked.includes(v.id);
            const box = el("input", { type: "checkbox", checked: on });
            const row = el("label", { className: "fg-vl-row", title: v.transcript },
                box,
                el("span", { className: "spk", textContent: on ? `#${speaker++}` : "" }),
                el("span", { className: "files", textContent: `${v.file} | ${v.transcript_file}` }),
                el("span", { className: "dur", textContent: v.duration != null ? `${v.duration.toFixed(1)}s` : "" }));
            box.addEventListener("change", () => {
                const now = selected().filter((id) => id !== v.id);
                if (box.checked) now.push(v.id);
                setSelected(now);
            });
            list.append(row);
        }
    }

    async function fetchVoices(rescan = false) {
        const folder = (folderW.value || "").trim();
        if (!folder) {
            voices = [];
            status.textContent = "";
            render();
            return;
        }
        status.textContent = rescan ? "Scanning…" : "Loading…";
        status.classList.remove("fg-vl-error");
        try {
            const resp = await api.fetchApi(`/fg/audio/voices?folder=${encodeURIComponent(folder)}&rescan=${rescan ? 1 : 0}`);
            const data = await resp.json();
            voices = data.voices || [];
            if (data.error) {
                status.textContent = data.error;
                status.classList.add("fg-vl-error");
            } else {
                const when = data.scanned ? new Date(data.scanned * 1000).toLocaleString() : "";
                status.textContent = `${voices.length} voice${voices.length === 1 ? "" : "s"}${when ? ` · scanned ${when}` : ""}`;
            }
        } catch (e) {
            voices = [];
            status.textContent = `Couldn't read the folder: ${e}`;
            status.classList.add("fg-vl-error");
        }
        render();
        requestAnimationFrame(() => {
            node.setSize([node.size[0], Math.max(node.size[1], node.computeSize()[1])]);
            node.setDirtyCanvas(true, true);
        });
    }

    all.addEventListener("click", () => setSelected(voices.map((v) => v.id)));
    none.addEventListener("click", () => setSelected([]));

    let timer = null;
    const folderCallback = folderW.callback;
    folderW.callback = function () {
        const r = folderCallback?.apply(this, arguments);
        clearTimeout(timer);
        timer = setTimeout(() => fetchVoices(false), 400);
        return r;
    };

    node.addWidget("button", "⟳ Re-scan folder", null, () => fetchVoices(true), { serialize: false });

    // ⟳ in the title bar too
    const TITLE_BTN = 22;
    const onDrawForeground = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
        const r = onDrawForeground?.apply(this, arguments);
        if (this.flags?.collapsed) return r;
        const h = window.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
        ctx.save();
        ctx.font = "15px sans-serif";
        ctx.fillStyle = "#bbb";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("⟳", this.size[0] - TITLE_BTN / 2 - 2, -h / 2);
        ctx.restore();
        return r;
    };
    const onMouseDown = node.onMouseDown;
    node.onMouseDown = function (e, pos) {
        if (pos[1] < 0 && pos[0] > this.size[0] - TITLE_BTN - 4) {
            fetchVoices(true);
            return true;
        }
        return onMouseDown?.apply(this, arguments);
    };

    // after a run, tick the voices actually used (a JSON dialogue chooses its own)
    const onExecuted = node.onExecuted;
    node.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        const used = message?.voices_used;
        if (Array.isArray(used) && used.length) setSelected([...new Set([...selected(), ...used.map(String)])]);
    };

    node.fgVoiceLibrary = { refresh: () => fetchVoices(false) };
    fetchVoices(false);
}

app.registerExtension({
    name: "FG.Audio.VoiceLibrary",

    async nodeCreated(node) {
        if (node.comfyClass === NODE_CLASS) setupLibrary(node);
    },

    // saved widget values (folder, ticks) land after nodeCreated
    async loadedGraphNode(node) {
        if (node.comfyClass === NODE_CLASS) node.fgVoiceLibrary?.refresh();
    },
});
