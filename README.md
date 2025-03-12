# ComfyUI-DSD

An Unofficial ComfyUI custom node package that integrates [Diffusion Self-Distillation (DSD)](https://github.com/primecai/diffusion-self-distillation) for zero-shot customized image generation.

DSD is a model for subject-preserving image generation that allows you to create images of a specific subject in novel contexts without per-instance tuning.

## Features

- Subject-preserving image generation using DSD model
- Gemini API prompt enhancement
- Direct model download from Hugging Face
- Fine-grained control over generation parameters

## Installation

1. Clone this repository into your ComfyUI custom_nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/irreveloper/ComfyUI-DSD.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Get the model files (two options):
   - **Option 1**: Use the `DSD Model Downloader` node in ComfyUI to automatically download the model
   - **Option 2**: Download manually from [Hugging Face](https://huggingface.co/primecai/dsd_model) or [Google Drive](https://drive.google.com/drive/folders/1VStt7J2whm5RRloa4NK1hGTHuS9WiTfO?usp=sharing)

   The model files will be stored in:
   - `ComfyUI/models/dsd_model/transformer/` (for transformer files)
   - `ComfyUI/models/dsd_model/pytorch_lora_weights.safetensors` (for LoRA file)

4. Restart ComfyUI

## Available Nodes

1. **DSD Model Downloader**: Automatically downloads the model from Hugging Face

2. **DSD Model Loader**: Loads a pre-downloaded model

3. **DSD Model Selector**: Helps select models from local directories

4. **DSD Gemini Prompt Enhancer**: Uses Google's Gemini API to enhance prompts for better image generation results. The API key can be provided in two ways:
   - As an input parameter to the node (not recommended for sharing workflows)
   - Through the `GEMINI_API_KEY` environment variable (strongly recommended)
   
   Note: To use the enhanced prompts, make sure to enable the `use_gemini_prompt` option on the DSD Image Generator node. If you don't enter a API Key it will be skipped automatically.

5. **DSD Image Generator**: Generates images with the DSD model

## Basic Workflow

![Sample1](examples/screenshot.png)

![Sample2](examples/screenshot-2.png)


## Troubleshooting

- **Memory Issues**: Try reducing precision (use bfloat16), lower resolution, or fewer steps
- **Gemini API**: Ensure you have a valid API key (can be set via GEMINI_API_KEY environment variable)
- **Model Loading**: If you see errors, try using the Model Downloader node to re-download files

## Examples

Check the `examples` directory for sample workflows. 