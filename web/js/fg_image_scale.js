/**
 * Farrenzo's Garbage — Image Scale widget control
 *
 * Two jobs:
 *   1. The three "enable_*" switches are mutually exclusive. Turning one on
 *      turns the other two off, and only the active mode's options stay
 *      visible. Structure borrowed from inpaint-cropandstitch/showcontrol.js.
 *      "use_model" nests inside enable_manual_size: it is NOT a fourth mode
 *      and takes no part in the exclusivity, it just gates its own three
 *      options behind its parent still being on.
 *   2. background_color gets a colour swatch you can click, plus right-click
 *      menu entries. ComfyUI still has no built-in colour widget type
 *      (Comfy-Org/ComfyUI#9531 is open), so this decorates the plain STRING
 *      widget instead of replacing it — if any of it fails on your frontend
 *      version, the text field still works exactly as before, and no extra
 *      widget is added so widgets_values stays the same length.
 */

const { app } = window.comfyAPI.app;
const NodeName = "Farrenzo.Garbage.ImageScale"


// ---------------------------------------------------------------------------
// Knobs
// ---------------------------------------------------------------------------
const CONFIG = {
    hideWidgets: true,   // false = grey the inactive options out instead of hiding them
    autoResize : true,   // shrink/grow the node when widgets appear or disappear
    legacyHide : false,  // set true only if hidden widgets still take up space on your frontend
};

const TARGET_CLASSES = ["FG_ImageScaler", "FG_ImageScale", "FG_Image_Scaler"];

const SWITCHES = {
    enable_round_to_multiple  : ["rounding", "round_to_multiple"],
    enable_scale_to_megapixels: ["megapixels", "resolution_steps"],
    enable_manual_size        : ["desired_width", "desired_height", "use_model"],
};

const SWITCH_NAMES = Object.keys(SWITCHES);

// Switches nested inside one of the three above. Their options show only when
// BOTH the parent switch and the sub-switch are on; the sub-switch itself is
// listed in its parent's deps, so it disappears with the parent. These are not
// mutually exclusive with anything -- they're a second level, not a fourth mode.
const SUB_SWITCHES = {
    use_model: {
        parent: "enable_manual_size",
        deps  : ["max_longest_length", "model_multiple", "upscale_model"],
    },
};

const SUB_SWITCH_NAMES = Object.keys(SUB_SWITCHES);

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
const findWidgetByName = (node, name) =>
    node.widgets ? node.widgets.find((w) => w.name === name) : null;

function isTargetNode(node) {
    if (node.comfyClass && TARGET_CLASSES.includes(node.comfyClass)) return true;
    // Fall back to duck typing so a rename of the mapping key doesn't break this.
    // Deliberately only the three top-level switches: a node from before the
    // upscale-model merge has no use_model widget and should still be managed.
    return SWITCH_NAMES.every((name) => !!findWidgetByName(node, name));
}

function getPropertyDescriptor(obj, prop) {
    let current = obj;
    while (current) {
        const desc = Object.getOwnPropertyDescriptor(current, prop);
        if (desc) return desc;
        current = Object.getPrototypeOf(current);
    }
    return null;
}

// ---------------------------------------------------------------------------
// Show / hide
// ---------------------------------------------------------------------------
function toggleWidget(node, widget, show) {
    if (!widget) return false;

    const hide = !show;
    const changed = (widget.hidden === true) !== hide;

    if (CONFIG.hideWidgets) {
        widget.hidden = hide;
        if (widget.element) widget.element.style.display = hide ? "none" : "";

        if (CONFIG.legacyHide) {
            if (hide) {
                if (widget.origType === undefined) {
                    widget.origType = widget.type;
                    widget.origComputeSize = widget.computeSize;
                }
                widget.type = "converted-widget";
                widget.computeSize = () => [0, -4];
            } else if (widget.origType !== undefined) {
                widget.type = widget.origType;
                widget.computeSize = widget.origComputeSize;
                delete widget.origType;
                delete widget.origComputeSize;
            }
        }
    }

    widget.disabled = hide;
    if (widget.options) widget.options.disabled = hide;
    if (widget._state) widget._state.disabled = hide;

    widget.linkedWidgets?.forEach((w) => toggleWidget(node, w, show));
    return changed;
}

function refreshSize(node) {
    if (!CONFIG.autoResize) return;
    const computed = node.computeSize?.();
    if (!computed) return;
    node.setSize([Math.max(node.size[0], computed[0]), computed[1]]);
}

// ---------------------------------------------------------------------------
// The actual rule
// ---------------------------------------------------------------------------
function updateNode(node, triggerName) {
    if (node.__fgScaleBusy) return;   // re-entry guard: we flip switch values below
    node.__fgScaleBusy = true;

    let layoutChanged = false;
    try {
        const switches = SWITCH_NAMES.map((n) => findWidgetByName(node, n)).filter(Boolean);

        // The switch the user just touched wins.
        if (triggerName && SWITCH_NAMES.includes(triggerName)) {
            const trigger = findWidgetByName(node, triggerName);
            if (trigger && trigger.value === true) {
                for (const other of switches) {
                    if (other.name !== triggerName && other.value === true) other.value = false;
                }
            }
        }

        // Anything still doubled up (loaded workflow, API-authored graph): first one wins,
        // same tie-break order Python uses.
        const on = switches.filter((w) => w.value === true);
        if (on.length > 1) {
            for (const extra of on.slice(1)) extra.value = false;
        }

        for (const name of SWITCH_NAMES) {
            const sw = findWidgetByName(node, name);
            const show = !!(sw && sw.value === true);
            for (const dep of SWITCHES[name]) {
                layoutChanged = toggleWidget(node, findWidgetByName(node, dep), show) || layoutChanged;
            }
        }

        // Second level, after the first: the parent loop above has already
        // shown or hidden the sub-switch widget itself, and a sub-switch's
        // options are only reachable when its parent survived that pass.
        for (const subName of SUB_SWITCH_NAMES) {
            const { parent, deps } = SUB_SWITCHES[subName];
            const sub = findWidgetByName(node, subName);
            if (!sub) continue;                      // pre-merge node, nothing to do
            const parentSwitch = findWidgetByName(node, parent);
            const parentOn = !!(parentSwitch && parentSwitch.value === true);
            const show = parentOn && sub.value === true;
            for (const dep of deps) {
                layoutChanged = toggleWidget(node, findWidgetByName(node, dep), show) || layoutChanged;
            }
        }
    } finally {
        node.__fgScaleBusy = false;
    }

    if (layoutChanged) refreshSize(node);
    node.setDirtyCanvas?.(true, true);
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
function hookWidget(node, widget) {
    if (!widget || widget.__fgHooked) return;
    widget.__fgHooked = true;

    const descriptor = getPropertyDescriptor(widget, "value");
    let stored = widget.value;

    Object.defineProperty(widget, "value", {
        get() {
            return descriptor && descriptor.get ? descriptor.get.call(this) : stored;
        },
        set(newValue) {
            if (descriptor && descriptor.set) descriptor.set.call(this, newValue);
            else stored = newValue;
            updateNode(node, this.name);
        },
        configurable: true,
        enumerable: true,
    });

    const origCallback = widget.callback;
    widget.callback = function () {
        const result = origCallback?.apply(this, arguments);
        updateNode(node, this.name);
        return result;
    };
}

app.registerExtension({
    name: NodeName,

    nodeCreated(node) {
        if (!isTargetNode(node)) return;

        const origOnConfigure = node.onConfigure;
        node.onConfigure = function () {
            const result = origOnConfigure?.apply(this, arguments);
            updateNode(this);
            return result;
        };

        for (const name of SWITCH_NAMES) hookWidget(node, findWidgetByName(node, name));
        // Sub-switches need the same value hook, otherwise their options only
        // appear after some unrelated widget happens to trigger a refresh.
        for (const name of SUB_SWITCH_NAMES) hookWidget(node, findWidgetByName(node, name));

        updateNode(node);
    },

    loadedGraphNode(node) {
        if (isTargetNode(node)) updateNode(node);
    },
});
