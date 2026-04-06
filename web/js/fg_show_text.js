import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
    name: "ShowText",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "FG_ShowText") return;

        function populate(text) {
            if (this.widgets) {
                const pos = this.widgets.findIndex((w) => w.name === "display_text");
                if (pos !== -1) {
                    for (let i = pos; i < this.widgets.length; i++) {
                        this.widgets[i].onRemove?.();
                    }
                    this.widgets.length = pos;
                }
            }

            // Flatten if nested (handles both onExecuted and onConfigure formats)
            let flatText = text;
            if (Array.isArray(text) && text.length > 0 && Array.isArray(text[0])) {
                flatText = text[0];  // Unwrap [[values]] to [values]
            }

            for (const item of flatText) {
                const w = ComfyWidgets["STRING"](this, "display_text", ["STRING", { multiline: true }], app).widget;
                w.inputEl.readOnly = true;
                w.inputEl.style.opacity = 0.6;
                w.value = item;
            }

            // Store the text for the copy button
            this._displayText = Array.isArray(flatText) ? flatText.join("\n") : String(flatText);

            requestAnimationFrame(() => {
                const sz = this.computeSize();
                if (sz[0] < this.size[0]) {
                    sz[0] = this.size[0];
                }
                if (sz[1] < this.size[1]) {
                    sz[1] = this.size[1];
                }
                this.onResize?.(sz);
                app.graph.setDirtyCanvas(true, false);
            });
        }

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            populate.call(this, message.text);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            if (this.widgets_values?.length) {
                // widgets_values is [[values]] format from Python
                populate.call(this, this.widgets_values);
            }
        };

        // Draw copy button on node
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            
            if (!this._displayText) return;

            // ===========================================
            // ADJUST THESE VALUES TO POSITION THE BUTTON
            // ===========================================
            const buttonSize = 24;      // Size of the button
            const fromRight = 42;       // Distance from right edge
            const fromTop = 8;          // Distance from top of node (below title)
            
            const x = this.size[0] - buttonSize - fromRight;
            const y = fromTop;

            this._copyButtonBounds = { x, y, width: buttonSize, height: buttonSize };

            // DEBUG: Log position once
            if (!this._debugLogged) {
                console.log(`Button position: x=${x}, y=${y}, nodeWidth=${this.size[0]}, nodeHeight=${this.size[1]}`);
                this._debugLogged = true;
            }

            // Draw button background
            if (this._copyFeedback) {
                ctx.fillStyle = "rgba(80, 180, 80, 0.9)";
            } else {
                ctx.fillStyle = this._copyButtonHover ? "rgba(100, 100, 100, 0.8)" : "rgba(60, 60, 60, 0.8)";
            }
            ctx.beginPath();
            ctx.roundRect(x, y, buttonSize, buttonSize, 4);
            ctx.fill();

            // Draw icon
            ctx.font = "14px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillStyle = this._copyButtonHover || this._copyFeedback ? "#fff" : "#ccc";
            ctx.fillText(this._copyFeedback ? "✓" : "📋", x + buttonSize / 2, y + buttonSize / 2 + 1);
        };

        // Handle mouse move for hover effect
        const onMouseMove = nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove = function (e, localPos) {
            onMouseMove?.apply(this, arguments);
            
            if (this._copyButtonBounds) {
                const { x, y, width, height } = this._copyButtonBounds;
                const wasHover = this._copyButtonHover;
                this._copyButtonHover = (
                    localPos[0] >= x && localPos[0] <= x + width &&
                    localPos[1] >= y && localPos[1] <= y + height
                );
                
                if (wasHover !== this._copyButtonHover) {
                    this.setDirtyCanvas(true, false);
                }
            }
        };

        // Handle click on copy button
        const onMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (e, localPos, graphCanvas) {
            if (this._copyButtonBounds && this._displayText) {
                const { x, y, width, height } = this._copyButtonBounds;
                
                if (localPos[0] >= x && localPos[0] <= x + width &&
                    localPos[1] >= y && localPos[1] <= y + height) {
                    
                    // Copy to clipboard
                    navigator.clipboard.writeText(this._displayText).then(() => {
                        this._copyFeedback = true;
                        this.setDirtyCanvas(true, false);
                        
                        setTimeout(() => {
                            this._copyFeedback = false;
                            this.setDirtyCanvas(true, false);
                        }, 500);
                        
                        console.log("Copied to clipboard:", this._displayText.substring(0, 50) + "...");
                    }).catch(err => {
                        console.error("Failed to copy:", err);
                    });
                    
                    return true;
                }
            }
            
            return onMouseDown?.apply(this, arguments);
        };
    },
});