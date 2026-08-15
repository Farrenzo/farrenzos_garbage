const { app } = window.comfyAPI.app;


import GLOBAL_SETTINGS from './_fg_settings.js';
const ASPECT_RATIOS = GLOBAL_SETTINGS.aspect_ratios
// Default selections per orientation
const DEFAULTS = {
    "Manual"     :"Manual",
    "Square"     :"1:1 | 64 | 1024×1024",
    "Horizontal" :"16:9 | 64 | 1536×896",
    "Vertical"   :"9:16 | 64 | 896×1536",
};

function setupNode(node) {
    // Find our widgets
    const orientationWidget = node.widgets.find(w => w.name === "orientation");
    const dimensionsWidget = node.widgets.find(w => w.name === "dimensions");
    const widthWidget = node.widgets.find(w => w.name === "width");
    const heightWidget = node.widgets.find(w => w.name === "height");
    
    if (!orientationWidget || !dimensionsWidget || !widthWidget || !heightWidget) {
        return;
    }

    // Function to update width/height from selected dimensions
    const updateWidthHeight = (dimensionKey) => {
        if (dimensionKey === "Manual") {
            return; // Don't auto-populate in manual mode
        }
        
        // Find the dimensions in our lookup
        for (const [orient, dims] of Object.entries(ASPECT_RATIOS)) {
            if (dims[dimensionKey]) {
                const [w, h] = dims[dimensionKey];
                widthWidget.value = w;
                heightWidget.value = h;
                break;
            }
        }
    };

    // Function to update dimensions dropdown based on orientation
    const updateDimensions = (orientation) => {
        let options;
        
        if (orientation === "Manual") {
            options = ["Manual"];
        } else {
            options = Object.keys(ASPECT_RATIOS[orientation] || {});
        }
        
        dimensionsWidget.options.values = options;
        
        // If current value not in new options, reset to default
        if (!options.includes(dimensionsWidget.value)) {
            dimensionsWidget.value = DEFAULTS[orientation] || options[0];
        }
        
        // Update width/height to match new selection
        updateWidthHeight(dimensionsWidget.value);
        
        // Trigger redraw
        node.setDirtyCanvas(true, true);
    };

    // Store original callbacks
    const originalOrientationCallback = orientationWidget.callback;
    const originalDimensionsCallback = dimensionsWidget.callback;
    
    // Override orientation widget callback
    orientationWidget.callback = function(value) {
        updateDimensions(value);
        if (originalOrientationCallback) {
            originalOrientationCallback.call(this, value);
        }
    };
    
    // Override dimensions widget callback to update width/height
    dimensionsWidget.callback = function(value) {
        updateWidthHeight(value);
        node.setDirtyCanvas(true, true);
        if (originalDimensionsCallback) {
            originalDimensionsCallback.call(this, value);
        }
    };

    // Initialize with current orientation
    updateDimensions(orientationWidget.value);
}

app.registerExtension({
    name: "AspectRatioLatentImage.Dynamic",
    
    async nodeCreated(node) {
        if (node.comfyClass !== "FG_EmptyLatent") {
            return;
        }
        setupNode(node);
    },

    async loadedGraphNode(node) {
        if (node.comfyClass !== "FG_EmptyLatent") {
            return;
        }

        // Small delay to ensure widgets are ready
        setTimeout(() => {
            setupNode(node);
            
            // Restore the dimensions options for the saved orientation
            const orientationWidget = node.widgets.find(w => w.name === "orientation");
            const dimensionsWidget = node.widgets.find(w => w.name === "dimensions");
            
            if (orientationWidget && dimensionsWidget) {
                const orientation = orientationWidget.value;
                let options;
                
                if (orientation === "Manual") {
                    options = ["Manual"];
                } else {
                    options = Object.keys(ASPECT_RATIOS[orientation] || {});
                }
                
                dimensionsWidget.options.values = options;
            }
        }, 100);
    }
});