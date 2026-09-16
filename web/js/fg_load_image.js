const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;


const MARGIN = 10; // node-space px, matches litegraph widget margin
const HANDLE = 8; // node-space px hit radius for corner handles
const MIN_SEL = 6; // drags smaller than this (node-space px) clear the crop
const MIN_EDITOR_H = 80; // minimum height of the crop editor area
// LGraphNode.resizeHandleSize — the corner zone litegraph resizes from.
const RESIZE_ZONE = 15;

const DEBUG = false;
function dbg(...args) {
    if (DEBUG) console.log("[🗑️ Garbãƶe]", ...args);
}
dbg("extension v5 (canvas widget) loaded");

function parseImageValue(value) {
    if (!value) return null;
    let filename = String(value);
    let type = "input";
    const annotated = filename.match(/^(.*) \[(\w+)\]$/);
    if (annotated) {
        filename = annotated[1];
        type = annotated[2];
    }
    let subfolder = "";
    const slash = filename.lastIndexOf("/");
    if (slash >= 0) {
        subfolder = filename.slice(0, slash);
        filename = filename.slice(slash + 1);
    }
    return { filename, type, subfolder };
}

app.registerExtension({
    name: "Farrenzo.GarbageLoadImage",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FG_LoadImage") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;
            const imageWidget = node.widgets.find((w) => w.name === "image");
            const cropWidget = node.widgets.find((w) => w.name === "crop");

            // The crop JSON widget is managed by the editor below.
            // widget.hidden hides it in the canvas renderer; options.hidden
            // hides it in the Nodes 2.0 (Vue) renderer.
            cropWidget.hidden = true;
            cropWidget.options = cropWidget.options || {};
            cropWidget.options.hidden = true;

            const isVueMode = () =>
                typeof LiteGraph !== "undefined" && !!LiteGraph.vueNodesMode;
            // UI metrics: Vue node cards render larger than graph units, so
            // chrome (text, handles) gets bumped up there.
            const ui = () =>
                isVueMode()
                    ? { font: 13, row: 18, handle: 12 }
                    : { font: 10, row: 14, handle: HANDLE };

            // node.imgs is not just the stock preview -- it is the ONLY thing
            // clipspace reads. copyToClipspace() copies it, and the mask editor
            // parses the /view?... query off imgs[i].src to build the
            // original_ref it posts to /upload/mask. That ref is what puts the
            // result in input/clipspace; blanking imgs left it empty, so masks
            // were written somewhere else entirely (and "Open in MaskEditor",
            // which is added conditionally on node.imgs, could vanish outright).
            //
            // So keep the DATA and kill the DRAWING: the stock preview is
            // painted by onDrawBackground, which is no-oped just below.
            const _imgs = [];
            Object.defineProperty(node, "imgs", {
                configurable: true,
                // undefined while empty -- callers test `if (node.imgs)` and an
                // empty array is truthy.
                get: () => (_imgs.length ? _imgs : undefined),
                set: (v) => {
                    _imgs.length = 0;
                    if (Array.isArray(v)) _imgs.push(...v);
                },
            });
            node.onDrawBackground = () => {}; // hide the stock preview
            node.setSizeForImage = () => {};  // ...and its auto-resize
            node.imageIndex = 0;              // clipspace's selectedIndex
            node.overIndex = null;

            const state = {
                img: null,
                rect: null, // normalized {x,y,w,h} or null = full image
                drag: null,
                box: null, // node-space letterbox of the image, set by draw()
            };

            try {
                const saved = cropWidget.value ? JSON.parse(cropWidget.value) : null;
                if (saved && saved.w > 0 && saved.h > 0) state.rect = saved;
            } catch (e) {
                state.rect = null;
            }

            function syncCrop() {
                let value = "";
                if (state.rect && state.rect.w > 0.001 && state.rect.h > 0.001) {
                    const r = state.rect;
                    // Treat a selection of (almost) everything as no crop.
                    if (!(r.x < 0.002 && r.y < 0.002 && r.w > 0.996 && r.h > 0.996)) {
                        const [px0, py0, px1, py1] = pixelBox();
                        value = JSON.stringify({
                            x: +r.x.toFixed(4),
                            y: +r.y.toFixed(4),
                            w: +r.w.toFixed(4),
                            h: +r.h.toFixed(4),
                            // The exact box travels alongside the normalized
                            // one: 1024 does not survive a round trip through
                            // 4 decimals of the image width, and the backend
                            // prefers px when it is present.
                            px: [px0, py0, px1, py1],
                        });
                    }
                }
                if (cropWidget.value !== value) {
                    cropWidget.value = value;
                    dbg("crop synced:", value || "(cleared)");
                }
            }

            function previewHeight(width) {
                if (!state.img) return 100;
                // Exact aspect fit so the image always spans the full width.
                return Math.round(width * (state.img.height / state.img.width));
            }

            // Mirror the backend's _parse_crop: an exact box set by a typed
            // size wins, otherwise round the normalized rect.
            function pixelBox() {
                const iw = state.img.width;
                const ih = state.img.height;
                const r = state.rect;
                let x0, y0, x1, y1;
                if (r.px) {
                    [x0, y0, x1, y1] = r.px.map((v) => Math.round(v));
                } else {
                    x0 = Math.round(r.x * iw);
                    y0 = Math.round(r.y * ih);
                    x1 = Math.round((r.x + r.w) * iw);
                    y1 = Math.round((r.y + r.h) * ih);
                }
                x0 = Math.max(0, Math.min(iw - 1, x0));
                y0 = Math.max(0, Math.min(ih - 1, y0));
                x1 = Math.max(x0 + 1, Math.min(iw, x1));
                y1 = Math.max(y0 + 1, Math.min(ih, y1));
                return [x0, y0, x1, y1];
            }

            function cropDims() {
                const [x0, y0, x1, y1] = pixelBox();
                return [x1 - x0, y1 - y0];
            }

            // --- typed crop size -------------------------------------------
            // crop_width / crop_height, 0 meaning "drag freely". One axis set
            // locks that axis; both set turn the selection into a fixed-size
            // box you click to position.
            const wWidget = node.widgets.find((w) => w.name === "crop_width");
            const hWidget = node.widgets.find((w) => w.name === "crop_height");

            function lockedSize() {
                const cw = Math.max(0, Math.round(Number(wWidget?.value) || 0));
                const ch = Math.max(0, Math.round(Number(hWidget?.value) || 0));
                return [cw, ch];
            }

            // Force the selection to the typed pixel size, keeping its centre
            // where it is -- or centring on (atX, atY) in node space if given.
            // Returns false when nothing is locked, so callers fall through to
            // the free-drag path.
            function applyLockedSize(atX, atY) {
                if (!state.img) return false;
                const [cw, ch] = lockedSize();
                if (!cw && !ch) return false;

                const iw = state.img.width;
                const ih = state.img.height;
                const cur = state.rect || { x: 0, y: 0, w: 1, h: 1 };
                const pw = Math.max(1, Math.min(iw, cw || Math.round(cur.w * iw)));
                const ph = Math.max(1, Math.min(ih, ch || Math.round(cur.h * ih)));

                let cx, cy;
                if (atX != null && state.box) {
                    // Node space -> image pixels.
                    const { bx, by, bw, bh } = state.box;
                    cx = ((atX - bx) / bw) * iw;
                    cy = ((atY - by) / bh) * ih;
                } else {
                    cx = (cur.x + cur.w / 2) * iw;
                    cy = (cur.y + cur.h / 2) * ih;
                }

                const x0 = Math.max(0, Math.min(iw - pw, Math.round(cx - pw / 2)));
                const y0 = Math.max(0, Math.min(ih - ph, Math.round(cy - ph / 2)));
                state.rect = {
                    x: x0 / iw,
                    y: y0 / ih,
                    w: pw / iw,
                    h: ph / ih,
                    px: [x0, y0, x0 + pw, y0 + ph],
                };
                return true;
            }

            // Returns [w, h] after the max_megapixels cap, or null if it
            // doesn't shrink this size.
            function cappedDims(w, h) {
                const mpWidget = node.widgets.find((x) => x.name === "max_megapixels");
                const mp = mpWidget ? Number(mpWidget.value) || 0 : 0;
                if (mp <= 0) return null;
                const target = mp * 1024 * 1024;
                if (w * h <= target) return null;
                const s = Math.sqrt(target / (w * h));
                return [Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s))];
            }

            function hitTest(px, py) {
                if (!state.rect || !state.box) return { mode: "new" };
                const [lkw, lkh] = lockedSize();
                // Both axes locked: there is nothing to resize, so the corners
                // become part of the move target rather than dragging the
                // selection off its typed dimensions.
                const fullyLocked = lkw > 0 && lkh > 0;
                const handle = ui().handle;
                const { bx, by, bw, bh } = state.box;
                const sx = bx + state.rect.x * bw;
                const sy = by + state.rect.y * bh;
                const sw = state.rect.w * bw;
                const sh = state.rect.h * bh;
                const corners = {
                    nw: [sx, sy], ne: [sx + sw, sy],
                    sw: [sx, sy + sh], se: [sx + sw, sy + sh],
                };
                if (!fullyLocked) {
                    for (const [name, [cx, cy]] of Object.entries(corners)) {
                        if (Math.abs(px - cx) <= handle && Math.abs(py - cy) <= handle) {
                            return { mode: "resize", corner: name };
                        }
                    }
                }
                if (px >= sx && px <= sx + sw && py >= sy && py <= sy + sh) {
                    return { mode: "move", offX: px - sx, offY: py - sy };
                }
                return { mode: "new" };
            }

            // The height litegraph allocated to this widget. Kept here (not
            // read back off the widget) because the computedHeight property
            // below reports a SHORTER box to litegraph's hit test, and
            // drawing must use the real allocation.
            let allocHeight;

            // The editor's real box height. This widget is the node's last
            // one, so the node's own size is the truth: deriving from it means
            // a stale allocation can never leave the preview drawn (or
            // hit-tested) at the wrong size. Falls back to the allocation.
            function boxHeight(widget, widgetY, fallback) {
                if (isVueMode()) return fallback;
                const nodeH = node.size?.[1];
                const visible = node.widgets?.filter((w) => !w.hidden);
                const isLast =
                    !!visible && visible[visible.length - 1] === widget;
                if (nodeH == null || widgetY == null || !isLast) return fallback;
                return Math.max(MIN_EDITOR_H, nodeH - widgetY);
            }

            const editor = {
                name: "crop_editor",
                type: "Garbãƶe",
                value: "",
                serialize: false,
                options: { serialize: false },

                // No computeSize: with computeLayoutSize the editor becomes a
                // "growable" widget in the canvas layout — it fills whatever
                // vertical space the node has, letterboxing the image, instead
                // of forcing the node's height to the image aspect.
                computeLayoutSize: function (n) {
                    if (isVueMode()) {
                        // Vue cards auto-size vertically: keep exact aspect.
                        const w = state.lastDrawW || (n?.size?.[0] ?? 200);
                        const h = previewHeight(Math.max(1, w - MARGIN * 2)) + ui().row + 8;
                        return { minHeight: h, maxHeight: h, minWidth: 0 };
                    }
                    return { minHeight: MIN_EDITOR_H, maxHeight: 100000, minWidth: 0 };
                },

                draw: function (ctx, _node, widgetWidth, y, H, lowQuality) {
                    // pasteFromClipspace() writes imageWidget.value straight in
                    // and does not always fire the widget callback, which used
                    // to leave the editor showing the pre-mask image.
                    // loadImage() updates lastLoaded up front, so this fires
                    // once per change rather than once per frame. The crop is
                    // deliberately kept: a mask round trip returns the same
                    // image at the same size, and re-selecting every time would
                    // be maddening.
                    if (imageWidget.value !== lastLoaded) loadImage();
                    const u = ui();
                    const h = boxHeight(this, y, allocHeight ?? H) - 8;
                    const x = MARGIN;
                    // In canvas mode the width param can lag the node during
                    // interactive resizing — trust the node's actual width
                    // when smaller. In Vue mode widgetWidth is the card's CSS
                    // width and node.size is unrelated, so use it as-is.
                    const nodeW = _node?.size?.[0];
                    const effWidth =
                        !isVueMode() && nodeW ? Math.min(widgetWidth, nodeW) : widgetWidth;
                    state.lastDrawW = effWidth;
                    const w = effWidth - MARGIN * 2;
                    const imgAreaH = Math.max(1, h - u.row);

                    ctx.save();

                    if (!state.img) {
                        ctx.fillStyle = "#00000033";
                        ctx.fillRect(x, y, w, h);
                        ctx.fillStyle = "#888";
                        ctx.font = `${u.font + 2}px sans-serif`;
                        ctx.textAlign = "center";
                        ctx.textBaseline = "middle";
                        ctx.fillText("no image", x + w / 2, y + h / 2);
                        ctx.restore();
                        return;
                    }

                    const scale = Math.min(w / state.img.width, imgAreaH / state.img.height);
                    const bw = state.img.width * scale;
                    const bh = state.img.height * scale;
                    const bx = x + (w - bw) / 2;
                    const by = y + (imgAreaH - bh) / 2;
                    state.box = { bx, by, bw, bh };
                    ctx.drawImage(state.img, bx, by, bw, bh);

                    if (state.rect && !lowQuality) {
                        const sx = bx + state.rect.x * bw;
                        const sy = by + state.rect.y * bh;
                        const sw = state.rect.w * bw;
                        const sh = state.rect.h * bh;

                        // Dim everything outside the selection.
                        ctx.beginPath();
                        ctx.rect(bx, by, bw, bh);
                        ctx.rect(sx, sy, sw, sh);
                        ctx.fillStyle = "rgba(0,0,0,0.55)";
                        ctx.fill("evenodd");

                        ctx.strokeStyle = "#4af";
                        ctx.lineWidth = 1;
                        ctx.strokeRect(sx, sy, sw, sh);
                        ctx.fillStyle = "#4af";
                        for (const [hx, hy] of [
                            [sx, sy], [sx + sw, sy], [sx, sy + sh], [sx + sw, sy + sh],
                        ]) {
                            ctx.fillRect(hx - 2.5, hy - 2.5, 5, 5);
                        }

                        // Crop dimensions above the selection.
                        ctx.font = `${u.font}px sans-serif`;
                        ctx.textAlign = "left";
                        ctx.textBaseline = "alphabetic";
                        const pillH = u.font + 2;
                        // Pills centered on the crop box, kept inside the
                        // image. Segments are [text, color] pairs.
                        const drawPill = (segments, ty) => {
                            const widths = segments.map((s) => ctx.measureText(s[0]).width);
                            const tw = widths.reduce((a, b) => a + b, 0);
                            const tx = Math.max(
                                bx,
                                Math.min(sx + (sw - tw - 6) / 2, bx + bw - tw - 6)
                            );
                            ctx.fillStyle = "rgba(0,0,0,0.6)";
                            ctx.fillRect(tx, ty - pillH + 3, tw + 6, pillH);
                            let cx = tx + 3;
                            for (const [i, [text, color]] of segments.entries()) {
                                ctx.fillStyle = color;
                                ctx.fillText(text, cx, ty);
                                cx += widths[i];
                            }
                        };
                        const [pw, ph] = cropDims();
                        const [lkw, lkh] = lockedSize();
                        // Blue matches the selection outline, so a locked box
                        // reads as locked without another row of chrome.
                        drawPill(
                            [[`${pw} x ${ph}`, lkw || lkh ? "#4af" : "#fff"]],
                            sy > y + pillH + 2 ? sy - 3 : sy + pillH - 1
                        );
                        const capped = cappedDims(pw, ph);
                        if (capped) {
                            // Downscale result below the selection.
                            const belowY = sy + sh + pillH - 1;
                            const ty = belowY < y + imgAreaH - 2 ? belowY : sy + sh - 4;
                            drawPill(
                                [
                                    ["Downscaled To: ", "#aaa"],
                                    [`${capped[0]} x ${capped[1]}`, "#fff"],
                                ],
                                ty
                            );
                        }
                    }

                    if (!lowQuality) {
                        // Info row below the image, centered. Labels muted,
                        // dimension values in the default widget text color.
                        const lg = typeof LiteGraph !== "undefined" ? LiteGraph : {};
                        const textColor = lg.WIDGET_TEXT_COLOR || "#ddd";
                        const MUTED_ALPHA = 0.45;
                        const iw = state.img.width;
                        const ih = state.img.height;
                        // Segments: [text, muted?]
                        const segments = [
                            ["Full: ", true],
                            [`${iw} x ${ih}`, false],
                        ];
                        if (!state.rect) {
                            const capped = cappedDims(iw, ih);
                            if (capped) {
                                segments.push(
                                    ["   Downscaled To: ", true],
                                    [`${capped[0]} x ${capped[1]}`, false]
                                );
                            }
                        }
                        ctx.font = `${u.font}px sans-serif`;
                        ctx.textBaseline = "alphabetic";
                        ctx.textAlign = "left";
                        ctx.fillStyle = textColor;
                        const ty = y + h - 3;
                        const total = segments.reduce(
                            (sum, s) => sum + ctx.measureText(s[0]).width, 0
                        );
                        let cx = x + (w - total) / 2;
                        const prevAlpha = ctx.globalAlpha;
                        for (const [text, muted] of segments) {
                            ctx.globalAlpha = muted ? prevAlpha * MUTED_ALPHA : prevAlpha;
                            ctx.fillText(text, cx, ty);
                            cx += ctx.measureText(text).width;
                        }
                        ctx.globalAlpha = prevAlpha;
                    }

                    ctx.restore();
                },

                mouse: function (event, pos, _node) {
                    if (!state.img || !state.box) return false;
                    const t = event.type;
                    const px = pos[0];
                    const py = pos[1];
                    const { bx, by, bw, bh } = state.box;
                    const clampX = (v) => Math.max(bx, Math.min(bx + bw, v));
                    const clampY = (v) => Math.max(by, Math.min(by + bh, v));

                    if (t === "pointerdown" || t === "mousedown") {
                        // Ignore clicks on the letterbox area outside the
                        // image — only the image itself is interactive.
                        if (px < bx || px > bx + bw || py < by || py > by + bh) {
                            return false;
                        }
                        const hit = hitTest(px, py);
                        const [lkw, lkh] = lockedSize();
                        if (lkw > 0 && lkh > 0 && hit.mode !== "move") {
                            // Fixed size: a click anywhere drops the box centred
                            // there and goes straight into a move drag, so the
                            // typed dimensions can't be dragged away from.
                            applyLockedSize(px, py);
                            state.drag = {
                                mode: "move",
                                offX: px - (bx + state.rect.x * bw),
                                offY: py - (by + state.rect.y * bh),
                                startX: px,
                                startY: py,
                                moved: false,
                            };
                            if (event.target?.style) event.target.style.cursor = "grabbing";
                            syncCrop();
                            this.triggerDraw?.();
                            return true;
                        }
                        state.drag = { ...hit, startX: px, startY: py, moved: false };
                        const el = event.target;
                        if (el?.style) {
                            el.style.cursor =
                                state.drag.mode === "move"
                                    ? "grabbing"
                                    : state.drag.mode === "resize"
                                        ? (state.drag.corner === "nw" || state.drag.corner === "se"
                                            ? "nwse-resize"
                                            : "nesw-resize")
                                        : "crosshair";
                        }
                        this.triggerDraw?.();
                        return true;
                    }

                    const drag = state.drag;
                    if (!drag) return false;

                    if (t === "pointermove" || t === "mousemove") {
                        if (Math.abs(px - drag.startX) + Math.abs(py - drag.startY) > 2) {
                            drag.moved = true;
                        }
                        if (drag.mode === "new") {
                            const x0 = clampX(Math.min(drag.startX, px));
                            const y0 = clampY(Math.min(drag.startY, py));
                            const x1 = clampX(Math.max(drag.startX, px));
                            const y1 = clampY(Math.max(drag.startY, py));
                            if (x1 - x0 >= MIN_SEL && y1 - y0 >= MIN_SEL) {
                                state.rect = {
                                    x: (x0 - bx) / bw,
                                    y: (y0 - by) / bh,
                                    w: (x1 - x0) / bw,
                                    h: (y1 - y0) / bh,
                                };
                            }
                        } else if (drag.mode === "move" && state.rect) {
                            let nx = (clampX(px - drag.offX) - bx) / bw;
                            let ny = (clampY(py - drag.offY) - by) / bh;
                            nx = Math.max(0, Math.min(1 - state.rect.w, nx));
                            ny = Math.max(0, Math.min(1 - state.rect.h, ny));
                            state.rect.x = nx;
                            state.rect.y = ny;
                            if (state.rect.px) {
                                // Move the exact box with it, then snap the
                                // normalized rect back onto it so what's drawn
                                // is what gets cropped.
                                const iw = state.img.width;
                                const ih = state.img.height;
                                const pw = state.rect.px[2] - state.rect.px[0];
                                const ph = state.rect.px[3] - state.rect.px[1];
                                const x0 = Math.max(0, Math.min(iw - pw, Math.round(nx * iw)));
                                const y0 = Math.max(0, Math.min(ih - ph, Math.round(ny * ih)));
                                state.rect.px = [x0, y0, x0 + pw, y0 + ph];
                                state.rect.x = x0 / iw;
                                state.rect.y = y0 / ih;
                            }
                        } else if (drag.mode === "resize" && state.rect) {
                            const r = state.rect;
                            let x0 = bx + r.x * bw;
                            let y0 = by + r.y * bh;
                            let x1 = x0 + r.w * bw;
                            let y1 = y0 + r.h * bh;
                            if (drag.corner.includes("w")) x0 = clampX(px);
                            if (drag.corner.includes("e")) x1 = clampX(px);
                            if (drag.corner.includes("n")) y0 = clampY(py);
                            if (drag.corner.includes("s")) y1 = clampY(py);
                            if (Math.abs(x1 - x0) >= MIN_SEL && Math.abs(y1 - y0) >= MIN_SEL) {
                                state.rect = {
                                    x: (Math.min(x0, x1) - bx) / bw,
                                    y: (Math.min(y0, y1) - by) / bh,
                                    w: Math.abs(x1 - x0) / bw,
                                    h: Math.abs(y1 - y0) / bh,
                                };
                            }
                        }
                        this.triggerDraw?.();
                        return true;
                    }

                    if (t === "pointerup" || t === "mouseup") {
                        // A plain click (no real drag) outside a fresh selection
                        // clears the crop back to the full image.
                        if (drag.mode === "new" && !drag.moved) {
                            state.rect = null;
                        }
                        // Re-snap after a free drag: with only one axis locked
                        // the other stays resizable, and this pulls the locked
                        // one back to the exact value on release. Guarded on
                        // state.rect so clearing the crop above still works.
                        if (state.rect) applyLockedSize();
                        state.drag = null;
                        if (event.target?.style) event.target.style.cursor = "";
                        syncCrop();
                        this.triggerDraw?.();
                        return true;
                    }
                    return false;
                },
            };
            const editorWidget = node.addCustomWidget(editor);

            // Hover cursors (canvas mode): resize arrows on handles, hand
            // over the selection, crosshair to draw, cell when a click would
            // clear the existing crop.
            // Outside the image: name the cursor litegraph would show rather
            // than blanking it. LGraphCanvas caches the last cursor it wrote
            // and skips redundant writes, so a bare "" from here desyncs that
            // cache and the resize cursor never appears again.
            function cursorOutside(px, py) {
                const w = node.size?.[0];
                const h = node.size?.[1];
                if (w == null || h == null) return "default";
                if (py <= h && py >= h - RESIZE_ZONE) {
                    if (px >= w - RESIZE_ZONE) return "nwse-resize";
                    if (px <= RESIZE_ZONE) return "nesw-resize";
                }
                return "default";
            }
            function cursorFor(px, py) {
                if (!state.img || !state.box) return cursorOutside(px, py);
                const { bx, by, bw, bh } = state.box;
                if (px < bx || px > bx + bw || py < by || py > by + bh) {
                    return cursorOutside(px, py);
                }
                const hit = hitTest(px, py);
                if (hit.mode === "resize") {
                    return hit.corner === "nw" || hit.corner === "se"
                        ? "nwse-resize"
                        : "nesw-resize";
                }
                if (hit.mode === "move") return "grab";
                const [lkw, lkh] = lockedSize();
                // Locked: a click out here repositions the box rather than
                // clearing it, so don't advertise not-allowed.
                if (lkw > 0 && lkh > 0) return "crosshair";
                // A click out here removes the existing crop (drag draws new).
                return state.rect ? "not-allowed" : "crosshair";
            }
            const prevMouseMove = node.onMouseMove;
            node.onMouseMove = function (e, pos, graphCanvas) {
                prevMouseMove?.apply(this, arguments);
                const el = graphCanvas?.canvas || app.canvas?.canvas;
                if (el && !state.drag) el.style.cursor = cursorFor(pos[0], pos[1]);
            };
            const prevMouseLeave = node.onMouseLeave;
            node.onMouseLeave = function () {
                prevMouseLeave?.apply(this, arguments);
                const el = app.canvas?.canvas;
                if (el) el.style.cursor = "";
            };

            // computedHeight serves two masters: litegraph's layout (which
            // sets it) and its widget hit test, LGraphNode.getWidgetOnPos.
            // A growable widget that fills the node reports a box covering
            // the bottom corners, and LGraphCanvas checks widgets BEFORE the
            // resize corner (both for the hover cursor and on pointerdown),
            // so the node becomes nearly impossible to resize. Report a box
            // that stops above the info row: that strip is text only, so
            // nothing interactive is given up, and the node's bottom edge
            // and corners go back to litegraph.
            //
            // In Vue (Nodes 2.0) mode the widget mirror prefers computedHeight
            // over computeSize — but computedHeight is a stale graph-units
            // value from the canvas-mode layout. Hide it there so the mirror
            // falls back to computeSize with the card's real CSS width.
            Object.defineProperty(editorWidget, "computedHeight", {
                configurable: true,
                get() {
                    if (isVueMode() || allocHeight == null) return undefined;
                    // Exactly the resize zone: the whole corner band is freed,
                    // and it lands below the image (the info row is drawn
                    // there), so the crop handles keep their grab radius.
                    // No lower clamp here: a floor could report MORE than the
                    // box at minimum node size, pushing the hit rect past the
                    // node's bottom edge — the very thing this avoids.
                    const box = boxHeight(this, this.y, allocHeight);
                    return Math.max(0, box - RESIZE_ZONE);
                },
                set(v) {
                    allocHeight = v;
                },
            });
            // Width shield: litegraph draws and hit-tests with
            // `widget.width || node.size[0]`, so a width left on the widget by
            // anything else would silently shrink both. Always defer to the node.
            Object.defineProperty(editorWidget, "width", {
                configurable: true,
                get: () => undefined,
                set: () => {},
            });

            // The upload widget publishes the picked image to the node-output
            // preview (shown on the Vue node card) — remove it so only the
            // crop editor displays the image.
            function clearStockPreview() {
                const wipe = () => {
                    try {
                        if (app.nodeOutputs) delete app.nodeOutputs[String(node.id)];
                    } catch (e) {}
                };
                wipe();
                requestAnimationFrame(() => requestAnimationFrame(wipe));
                setTimeout(wipe, 150);
            }

            const mpWidget = node.widgets.find((w) => w.name === "max_megapixels");
            if (mpWidget) {
                const prevMpCallback = mpWidget.callback;
                mpWidget.callback = function () {
                    const r = prevMpCallback?.apply(this, arguments);
                    editorWidget.triggerDraw?.();
                    node.setDirtyCanvas(true, true);
                    return r;
                };
            }

            for (const sizeWidget of [wWidget, hWidget]) {
                if (!sizeWidget) continue;
                const prevSizeCallback = sizeWidget.callback;
                sizeWidget.callback = function () {
                    const r = prevSizeCallback?.apply(this, arguments);
                    if (applyLockedSize()) {
                        syncCrop();
                    } else if (state.rect) {
                        // Lock cleared: drop the exact box but keep the
                        // selection, which is now freely draggable again.
                        delete state.rect.px;
                        syncCrop();
                    }
                    editorWidget.triggerDraw?.();
                    node.setDirtyCanvas(true, true);
                    return r;
                };
            }

            let loadSeq = 0;
            let lastLoaded = null;
            function loadImage(autoFit = false) {
                lastLoaded = imageWidget.value;
                const seq = ++loadSeq;
                const info = parseImageValue(imageWidget.value);
                if (!info) {
                    state.img = null;
                    node.setDirtyCanvas(true, true);
                    return;
                }
                const url = api.apiURL(
                    `/view?filename=${encodeURIComponent(info.filename)}` +
                    `&type=${info.type}&subfolder=${encodeURIComponent(info.subfolder)}` +
                    `&rand=${Math.random()}`
                );
                const img = new Image();
                img.onload = () => {
                    if (seq !== loadSeq) return; // superseded by a newer load
                    state.img = img;
                    // Clipspace reads the /view?... src off this.
                    node.imgs = [img];
                    dbg("image loaded:", info.filename, img.width + "x" + img.height);

                    // Re-apply a typed size to the new image, but only when it
                    // doesn't already match -- otherwise restoring a saved
                    // workflow would re-centre a crop that was already correct.
                    const [lkw, lkh] = lockedSize();
                    if (lkw || lkh) {
                        const [curW, curH] = state.rect ? cropDims() : [0, 0];
                        if ((lkw && curW !== lkw) || (lkh && curH !== lkh)) {
                            if (applyLockedSize()) syncCrop();
                        }
                    }
                    if (autoFit && !isVueMode()) {
                        // Fit the node height to the image aspect once, when
                        // the image (first) loads; afterwards the user can
                        // resize freely and the editor letterboxes.
                        const minSize = node.computeSize();
                        const desired =
                            previewHeight(node.size[0] - MARGIN * 2) + ui().row + 8;
                        const height =
                            minSize[1] - MIN_EDITOR_H + Math.max(MIN_EDITOR_H, desired);
                        node.setSize([Math.max(node.size[0], minSize[0]), height]);
                    }
                    node.setDirtyCanvas(true, true);
                    editorWidget.triggerDraw?.();
                };
                img.onerror = () => {
                    if (seq !== loadSeq) return;
                    state.img = null;
                    node.setDirtyCanvas(true, true);
                    editorWidget.triggerDraw?.();
                };
                img.src = url;
                clearStockPreview();
            }

            const prevCallback = imageWidget.callback;
            imageWidget.callback = function () {
                const r = prevCallback?.apply(this, arguments);
                state.rect = null;
                syncCrop();
                // Keep the node's current size — the preview letterboxes.
                loadImage();
                return r;
            };

            // Workflow loading assigns widgets_values directly (no widget
            // callbacks) after onNodeCreated — re-read the restored values.
            const prevOnConfigure = node.onConfigure;
            node.onConfigure = function () {
                const r = prevOnConfigure?.apply(this, arguments);
                try {
                    const saved = cropWidget.value ? JSON.parse(cropWidget.value) : null;
                    state.rect = saved && saved.w > 0 && saved.h > 0 ? saved : null;
                } catch (e) {
                    state.rect = null;
                }
                dbg("configured; crop:", cropWidget.value || "(none)", "image:", imageWidget.value);
                loadImage();
                return r;
            };

            loadImage(true); // fresh node: fit to the default image
            return result;
        };
    },
});