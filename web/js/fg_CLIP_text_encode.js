const { app } = window.comfyAPI.app;

const WIDGET_HEIGHT = 55;

/* ------------------------------------------------------------------ *
 *  Selection / match highlighting
 *
 *  A <textarea> cannot style parts of its own text, so we draw a
 *  transparent "mirror" of the text on top of it and only paint the
 *  matched ranges. The mirror is a fixed-position, pointer-events:none
 *  overlay so ComfyUI's DOM-widget positioning is never touched.
 * ------------------------------------------------------------------ */

const HL_STYLE_ID = "fg-match-highlight-style";

// Everything that affects where a glyph lands, copied from the textarea.
const MIRROR_PROPS = [
    "fontFamily", "fontSize", "fontWeight", "fontStyle", "fontVariant", "fontStretch",
    "letterSpacing", "wordSpacing", "lineHeight", "textIndent", "textTransform",
    "textAlign", "direction", "whiteSpace", "wordBreak", "overflowWrap", "tabSize",
    "paddingTop", "paddingRight", "paddingBottom", "paddingLeft",
];

function ensureHighlightStyles() {
    if (document.getElementById(HL_STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = HL_STYLE_ID;
    style.textContent = `
.fg-hl-overlay {
    position: fixed;
    overflow: hidden;
    pointer-events: none;
    z-index: 9999;
}
.fg-hl-scaler {
    position: absolute;
    top: 0; left: 0;
    transform-origin: 0 0;
}
.fg-hl-mirror {
    position: absolute;
    box-sizing: border-box;
    margin: 0;
    border: 0;
    color: transparent;
    background: transparent;
}
.fg-hl-mirror mark {
    color: transparent;
    background: transparent;
    border-radius: 2px;
}
.fg-hl-mirror mark.fg-hl-match {
  background: rgba(217, 70, 160, 0.25);
  box-shadow: 0 0 6px rgba(217, 70, 160, 0.55);
  border-radius: 3px;
}
.fg-hl-mirror mark.fg-hl-self {
  background: rgba(217, 70, 160, 0.10);
  border-radius: 3px;
}`;
    document.head.appendChild(style);
}

function escapeHtml(text) {
    return text.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

/**
 * Highlight every literal occurrence of the selected substring.
 *
 * Matching is plain substring matching, so selecting "one," matches only
 * the occurrences followed by a comma, while selecting "one" matches all
 * of them. No word-boundary or token logic is applied.
 *
 * @returns {() => void} dispose function
 */
function attachMatchHighlighter(textarea, options = {}) {
    const {
        minLength      = 2,     // ignore 1-char selections; they match everywhere
        caseSensitive  = false,
        trimSelection  = true,  // ignore whitespace dragged in at either end
        hideOnBlur     = true,  // set false to keep highlights after clicking away
        markSelf       = true,  // faintly mark the selected range itself
    } = options;

    ensureHighlightStyles();

    const overlay = document.createElement("div");
    overlay.className = "fg-hl-overlay";
    overlay.style.display = "none";

    const scaler = document.createElement("div");
    scaler.className = "fg-hl-scaler";

    const mirror = document.createElement("div");
    mirror.className = "fg-hl-mirror";

    scaler.appendChild(mirror);
    overlay.appendChild(scaler);
    document.body.appendChild(overlay);

    let rafId = 0;
    let active = false;

    function findRanges(haystack, needle) {
        const hay = caseSensitive ? haystack : haystack.toLowerCase();
        const pin = caseSensitive ? needle : needle.toLowerCase();
        const ranges = [];
        let i = hay.indexOf(pin);
        while (i !== -1) {
            ranges.push([i, i + pin.length]);
            i = hay.indexOf(pin, i + pin.length); // non-overlapping
        }
        return ranges;
    }

    function paint(text, ranges, selection) {
        let html = "";
        let cursor = 0;
        for (const [start, end] of ranges) {
            const isSelf = selection && start === selection[0] && end === selection[1];
            if (isSelf && !markSelf) {
                html += escapeHtml(text.slice(cursor, end));
                cursor = end;
                continue;
            }
            html += escapeHtml(text.slice(cursor, start));
            html += `<mark class="${isSelf ? "fg-hl-self" : "fg-hl-match"}">`
                 +  escapeHtml(text.slice(start, end))
                 +  `</mark>`;
            cursor = end;
        }
        // Trailing newline keeps the final line box measurable.
        mirror.innerHTML = html + escapeHtml(text.slice(cursor)) + "\n";
    }

    function schedule() {
        if (!rafId && active) rafId = requestAnimationFrame(sync);
    }

    // Keep the overlay glued to the textarea: position, canvas zoom, scroll.
    function sync() {
        rafId = 0;
        if (!active) return;

        if (!textarea.isConnected || textarea.offsetParent === null) return hide();

        const rect = textarea.getBoundingClientRect();
        if (rect.width < 1 || rect.height < 1) return hide();

        const cs    = getComputedStyle(textarea);
        const scale = textarea.offsetWidth ? rect.width / textarea.offsetWidth : 1;

        overlay.style.left         = `${rect.left}px`;
        overlay.style.top          = `${rect.top}px`;
        overlay.style.width        = `${rect.width}px`;
        overlay.style.height       = `${rect.height}px`;
        overlay.style.borderRadius = cs.borderRadius;

        scaler.style.transform = `scale(${scale})`;

        for (const prop of MIRROR_PROPS) mirror.style[prop] = cs[prop];
        // clientWidth = content + padding, minus any scrollbar, so the mirror
        // wraps exactly where the textarea wraps.
        mirror.style.width = `${textarea.clientWidth}px`;
        mirror.style.left  = `${parseFloat(cs.borderLeftWidth) - textarea.scrollLeft}px`;
        mirror.style.top   = `${parseFloat(cs.borderTopWidth) - textarea.scrollTop}px`;

        schedule();
    }

    function hide() {
        active = false;
        overlay.style.display = "none";
        if (rafId) {
            cancelAnimationFrame(rafId);
            rafId = 0;
        }
    }

    function update() {
        if (hideOnBlur && document.activeElement !== textarea) return hide();

        const text = textarea.value ?? "";
        let start  = textarea.selectionStart ?? 0;
        let end    = textarea.selectionEnd ?? 0;
        let term   = text.slice(start, end);

        if (trimSelection) {
            const lead = term.length - term.trimStart().length;
            term  = term.trim();
            start = start + lead;
            end   = start + term.length;
        }

        if (term.length < minLength) return hide();

        const ranges = findRanges(text, term);
        if (ranges.length < 2) return hide(); // only the selection itself matched

        paint(text, ranges, [start, end]);
        active = true;
        overlay.style.display = "";
        sync();
    }

    const onSelectionChange = () => {
        if (document.activeElement === textarea) update();
        else if (hideOnBlur) hide();
    };
    const onScroll = () => { if (active) sync(); };

    document.addEventListener("selectionchange", onSelectionChange);
    textarea.addEventListener("input",   update);
    textarea.addEventListener("mouseup", update);   // fallbacks for browsers that
    textarea.addEventListener("keyup",   update);   // are stingy with selectionchange
    textarea.addEventListener("scroll",  onScroll);
    if (hideOnBlur) textarea.addEventListener("blur", hide);

    return function dispose() {
        hide();
        document.removeEventListener("selectionchange", onSelectionChange);
        textarea.removeEventListener("input",   update);
        textarea.removeEventListener("mouseup", update);
        textarea.removeEventListener("keyup",   update);
        textarea.removeEventListener("scroll",  onScroll);
        textarea.removeEventListener("blur",    hide);
        overlay.remove();
    };
}

// Multiline widgets expose the element as inputEl (older) or element (newer).
function getTextArea(widget) {
    const el = widget?.inputEl || widget?.element;
    if (!el) return null;
    return el.tagName === "TEXTAREA" ? el : el.querySelector?.("textarea") ?? null;
}

function attachHighlightersToNode(node, widgetNames, attempt = 0) {
    node.__fgHighlightDisposers ??= [];
    let missing = false;

    for (const name of widgetNames) {
        const widget = node.widgets?.find((w) => w.name === name);
        if (!widget) continue;
        if (widget.__fgHighlighted) continue;

        const textarea = getTextArea(widget);
        if (!textarea) {
            missing = true;
            continue;
        }

        widget.__fgHighlighted = true;
        node.__fgHighlightDisposers.push(attachMatchHighlighter(textarea));
    }

    // The DOM element is created a tick or two after the widget in some
    // frontend versions.
    if (missing && attempt < 10) {
        setTimeout(() => attachHighlightersToNode(node, widgetNames, attempt + 1), 100);
        return;
    }

    if (!node.__fgHighlightCleanupHooked) {
        node.__fgHighlightCleanupHooked = true;
        const origOnRemoved = node.onRemoved;
        node.onRemoved = function () {
            for (const dispose of this.__fgHighlightDisposers ?? []) dispose();
            this.__fgHighlightDisposers = [];
            return origOnRemoved?.apply(this, arguments);
        };
    }
}

/* ------------------------------------------------------------------ *
 *  Existing widget visibility logic
 * ------------------------------------------------------------------ */

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

const HIGHLIGHT_WIDGETS = ["positive_prompt", "negative_prompt", "vl_instruction"];

app.registerExtension({
    name: "Farrenzo.DynamicCLIPTextEncode",

    async nodeCreated(node) {
        if (node.comfyClass !== "FG_CLIPTextEncode") return;
        attachHighlightersToNode(node, HIGHLIGHT_WIDGETS);
        setupNode(node);
    },

    async loadedGraphNode(node) {
        if (node.comfyClass !== "FG_CLIPTextEncode") return;
        setTimeout(() => {
            attachHighlightersToNode(node, HIGHLIGHT_WIDGETS);
            setupNode(node);
        }, 100);
    },
});