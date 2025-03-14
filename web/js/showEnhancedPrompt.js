import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Displays enhanced prompt on DSDGeminiPromptEnhancer node
app.registerExtension({
    name: "comfyui.dsd.showEnhancedPrompt",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DSDGeminiPromptEnhancer") {
            function normalizePrompt(prompt) {
                if (!prompt) return "";
                if (Array.isArray(prompt)) {
                    return prompt.join("");
                }
                return prompt;
            }

            function populateEnhancedPrompt(enhancedPrompt) {
                enhancedPrompt = normalizePrompt(enhancedPrompt);
                if (!enhancedPrompt) {
                    return;
                }

                try {
                    if (this.widgets) {
                        for (let i = 0; i < this.widgets.length; i++) {
                            if (this.widgets[i].name === "enhanced_prompt_display") {
                                this.widgets[i].onRemove?.();
                                this.widgets.splice(i, 1);
                                i--;
                            }
                        }
                    }

                    const textWidgetResult = ComfyWidgets["STRING"](this, "enhanced_prompt_text", ["STRING", { multiline: true }], app);
                    if (!textWidgetResult || !textWidgetResult.widget || !textWidgetResult.widget.inputEl) {
                        return;
                    }
                    const w = textWidgetResult.widget;
                    w.inputEl.readOnly = true;
                    w.inputEl.style.opacity = 0.8;
                    w.inputEl.style.backgroundColor = "#1e2124";
                    w.inputEl.style.color = "#9eec51";
                    w.name = "enhanced_prompt_display";
                    w.value = enhancedPrompt;

                    requestAnimationFrame(() => {
                        const sz = this.computeSize();
                        if (sz[0] < this.size[0]) sz[0] = this.size[0];
                        if (sz[1] < this.size[1]) sz[1] = this.size[1];
                        this.onResize?.(sz);
                        app.graph.setDirtyCanvas(true, false);
                    });
                } catch (error) {
                    // Silent error handling
                }
            }

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                try {
                    onExecuted?.apply(this, arguments);
                    if (message.enhanced_prompt) {
                        populateEnhancedPrompt.call(this, message.enhanced_prompt);
                    } else if (message.ui && message.ui.enhanced_prompt) {
                        populateEnhancedPrompt.call(this, message.ui.enhanced_prompt);
                    } else if (message.text) {
                        populateEnhancedPrompt.call(this, message.text);
                    }
                } catch (error) {
                    // Silent error handling
                }
            };
        }
    },
});

window.DSD_ENHANCED_PROMPT_LOADED = true;