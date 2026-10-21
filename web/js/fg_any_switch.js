const { app } = window.comfyAPI.app;

const NODE_CLASS = "FG_AnySwitch";
const MAX_PAIRS = 16; // must match MAX_PAIRS in fg_any_switch.py
const PAIR_RE = /^on_(true|false)_(\d+)$/;
const OUT_RE = /^out_(\d+)$/;

function getLink(node, id) {
  const g = node.graph ?? app.graph;
  return g?.getLink?.(id) ?? g?.links?.[id];
}

function highestUsedPair(node) {
  let hi = -1;
  for (const inp of node.inputs ?? []) {
    const m = PAIR_RE.exec(inp.name);
    if (m && inp.link != null) hi = Math.max(hi, +m[2]);
  }
  for (const out of node.outputs ?? []) {
    const m = OUT_RE.exec(out.name);
    if (m && out.links?.length) hi = Math.max(hi, +m[1]);
  }
  return hi;
}

function sync(node) {
  // used pairs + one empty spare
  const visible = Math.min(highestUsedPair(node) + 2, MAX_PAIRS);

  // Trim from the end only, so out_N stays at slot N+1 (matches RETURN_TYPES).
  for (let i = MAX_PAIRS - 1; i >= visible; i--) {
    for (const side of ["false", "true"]) {
      const idx = node.findInputSlot(`on_${side}_${i}`);
      if (idx !== -1) node.removeInput(idx);
    }
    const o = node.findOutputSlot(`out_${i}`);
    if (o !== -1) node.removeOutput(o);
  }

  // Grow in ascending order (appends land in the right place).
  for (let i = 0; i < visible; i++) {
    for (const side of ["true", "false"]) {
      if (node.findInputSlot(`on_${side}_${i}`) === -1) node.addInput(`on_${side}_${i}`, "*");
    }
    if (node.findOutputSlot(`out_${i}`) === -1) node.addOutput(`out_${i}`, "*");
  }

  // Label each input by what's wired into it, and retype each output to match its pair.
  for (let i = 0; i < visible; i++) {
    let type = "*";
    for (const side of ["true", "false"]) {
      const inp = node.inputs[node.findInputSlot(`on_${side}_${i}`)];
      if (!inp) continue;
      const link = inp.link != null ? getLink(node, inp.link) : null;
      const linkType = link?.type && link.type !== "*" ? link.type : null;
      inp.label = linkType ? `${linkType} (${side})` : `on_${side}_${i}`;
      if (linkType && type === "*") type = linkType; // true side wins
    }
    const out = node.outputs[node.findOutputSlot(`out_${i}`)];
    if (out) {
      out.type = type;
      out.label = type === "*" ? `out_${i}` : `${type}_${i}`;
    }
  }

  const s = node.computeSize();
  node.setSize([Math.max(node.size[0], s[0]), s[1]]);
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "Farrenzo.AnySwitch",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_CLASS) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      sync(this); // Python declares all 16 pairs; trim down to 1
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      requestAnimationFrame(() => sync(this)); // after links are restored
      return r;
    };

    const onConn = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const r = onConn?.apply(this, arguments);
      requestAnimationFrame(() => sync(this)); // don't mutate mid-callback
      return r;
    };
  },
});