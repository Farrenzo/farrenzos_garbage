const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const NODE_TYPE = "FG_NoisySeeder";

const SPECIAL_SEED_RANDOM = -1;
const SPECIAL_SEED_INCREMENT = -2;
const SPECIAL_SEED_DECREMENT = -3;
const SPECIAL_SEEDS = [SPECIAL_SEED_RANDOM, SPECIAL_SEED_INCREMENT, SPECIAL_SEED_DECREMENT];

// Matches the range new_random_seed() uses on the python side.
const RANDOM_MIN = 1;
const RANDOM_MAX = 1125899906842624;

const LAST_SEED_BUTTON_LABEL = "♻️ (Use Last Queued Seed)";

// Live instances, so the queue hook can find them without walking the graph
// (nodes inside subgraphs aren't in app.graph._nodes).
const SEEDER_NODES = new Set();

/** A fresh seed inside the range the python node uses. */
function generateRandomSeed() {
    return Math.floor(Math.random() * (RANDOM_MAX - RANDOM_MIN + 1)) + RANDOM_MIN;
}

/** Resolves -1 / -2 / -3 into a concrete seed. Anything else passes straight through. */
function getSeedToUse(node) {
    const inputSeed = Number(node.seedWidget.value);
    if (!SPECIAL_SEEDS.includes(inputSeed)) return inputSeed;

    const last = node.lastSeed;
    if (typeof last === "number" && !SPECIAL_SEEDS.includes(last)) {
        if (inputSeed === SPECIAL_SEED_INCREMENT) return last + 1;
        if (inputSeed === SPECIAL_SEED_DECREMENT) return last - 1;
    }
    return generateRandomSeed();
}

/** Prompt keys are "5" at the top level and "12:5" for a node inside a subgraph. */
function findApiNodeId(output, nodeId) {
    const id = String(nodeId);
    if (output[id]) return id;
    return Object.keys(output).find((key) => key.endsWith(`:${id}`)) ?? null;
}

/**
 * Swaps the special seed for a real one in the prompt and the serialized workflow,
 * after graphToPrompt() has run but before either is sent to the server. The widget
 * on the canvas is left alone, so "-1" stays "-1" while the metadata records the
 * seed that was actually used.
 */
function applySeedToPrompt(node, data) {
    const apiId = findApiNodeId(data?.output ?? {}, node.id);
    const inputs = apiId ? data.output[apiId]?.inputs : null;
    // Missing means muted, bypassed, or only part of the graph was queued.
    if (!inputs || inputs["seed"] === undefined) return;

    const seed = getSeedToUse(node);
    inputs["seed"] = seed;

    const workflowNode = data.workflow?.nodes?.find((n) => String(n.id) === String(node.id));
    if (workflowNode?.widgets_values) {
        workflowNode.widgets_values[node.widgets.indexOf(node.seedWidget)] = seed;
    }

    node.lastSeed = seed;
    if (seed !== Number(node.seedWidget.value)) {
        node.lastSeedButton.label = `♻️ ${seed}`;
        node.lastSeedButton.disabled = false;
    } else {
        node.lastSeedButton.label = LAST_SEED_BUTTON_LABEL;
        node.lastSeedButton.disabled = true;
    }
}

app.registerExtension({
    name: "Farrenzo.NoisySeeder",

    setup() {
        const originalQueuePrompt = api.queuePrompt;
        api.queuePrompt = async function (number, data) {
            for (const node of SEEDER_NODES) {
                applySeedToPrompt(node, data);
            }
            return originalQueuePrompt.apply(this, arguments);
        };
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            this.serialize_widgets = true;
            this.lastSeed = undefined;

            // Grab the seed widget and drop comfy's control_after_generate —
            // the special seeds below replace it.
            for (let i = this.widgets.length - 1; i >= 0; i--) {
                const w = this.widgets[i];
                if (w.name === "seed") {
                    this.seedWidget = w;
                    this.seedWidget.value = SPECIAL_SEED_RANDOM;
                } else if (w.name === "control_after_generate") {
                    this.widgets.splice(i, 1);
                }
            }

            this.addWidget("button", "🎲 Randomize Each Time", "", () => {
                this.seedWidget.value = SPECIAL_SEED_RANDOM;
            }, { serialize: false });

            this.addWidget("button", "🎲 New Fixed Random", "", () => {
                this.seedWidget.value = generateRandomSeed();
            }, { serialize: false });

            this.lastSeedButton = this.addWidget("button", "USE_LAST_SEED", "", () => {
                if (this.lastSeed == null) return;
                this.seedWidget.value = this.lastSeed;
                this.lastSeedButton.label = LAST_SEED_BUTTON_LABEL;
                this.lastSeedButton.disabled = true;
            }, { serialize: false });
            this.lastSeedButton.label = LAST_SEED_BUTTON_LABEL;
            this.lastSeedButton.disabled = true;

            SEEDER_NODES.add(this);
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            SEEDER_NODES.delete(this);
            onRemoved?.apply(this, arguments);
        };
    },
});