import os
import torch
import folder_paths
from PIL import Image
import numpy as np
import shutil
import requests
import json
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download, snapshot_download, HfHubHTTPError

from huggingface_hub import login

from .utils import get_model_path, get_lora_path, comfy_to_pil, pil_to_comfy, resize_and_center_crop, center_crop, pad_resize, fit_resize
from .dsd_imports import FluxConditionalPipeline, FluxTransformer2DConditionalModel, enhance_prompt, IMPORTS_AVAILABLE
from comfy.utils import ProgressBar
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Google Gemini API not available. Install with: pip install google-genai")

# Add support paths
custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Add ComfyUI models path
comfyui_models_path = os.path.join(os.path.dirname(custom_nodes_dir), "models")
os.makedirs(comfyui_models_path, exist_ok=True)
dsd_model_path = os.path.join(comfyui_models_path, "dsd_model")
os.makedirs(dsd_model_path, exist_ok=True)

# Register the dsd_model path with ComfyUI
folder_paths.add_model_folder_path("dsd_models", dsd_model_path)

class DSDModelLoader:
    """Loads the DSD (Diffusion Self-Distillation) model"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": ""}),
                "lora_path": ("STRING", {"default": ""}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "low_cpu_mem_usage": ("BOOLEAN", {"default": True, "tooltip": "Reduces CPU memory usage during model loading. Recommended for faster loading."}),
                "model_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "Offloads state dict to reduce memory usage during loading. May slow down inference speed."}), 
                "sequential_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "Enables sequential CPU offloading. Only use if low on VRAM. Significantly impacts inference speed."})
            }
        }
    
    RETURN_TYPES = ("DSD_MODEL",)
    RETURN_NAMES = ("dsd_model",)
    FUNCTION = "load_model"
    CATEGORY = "DSD"
    
    def load_model(self, model_path, lora_path, device, dtype, low_cpu_mem_usage, model_cpu_offload, sequential_cpu_offload):
        if not IMPORTS_AVAILABLE:
            raise ImportError("Could not import DSD modules. Make sure DSD project files (pipeline.py, transformer.py) are properly installed in the parent directory.")
        
        # Check if model_path is empty, use default path
        if not model_path:
            model_path = os.path.join(dsd_model_path, "transformer", "diffusion_pytorch_model.safetensors")
            print(f"Using default model path: {model_path}")
        
        # Check if lora_path is empty, use default path
        if not lora_path:
            lora_path = os.path.join(dsd_model_path, "pytorch_lora_weights.safetensors")
            print(f"Using default lora path: {lora_path}")
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please use DSDModelDownloader to download the model first.")
        
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found at {lora_path}. Please use DSDModelDownloader to download the model first.")

        print("Loading model...")
        # Convert dtype string to torch dtype
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }.get(dtype, torch.bfloat16)
        
        print("Loading transformer...")

        model_folder = os.path.dirname(model_path)
        # Load model with user-specified parameters
        transformer = FluxTransformer2DConditionalModel.from_pretrained(
            model_folder,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
        )
        
        

        print("Loading pipeline...")
        
        # Use the optimized from_pretrained method (which was monkey-patched in pipeline.py)
            # Use the optimized from_pretrained method
        pipe = FluxConditionalPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            transformer=transformer,
            torch_dtype=torch_dtype
            )
        
        # Access and modify scheduler configs
        pipe.scheduler.config.shift = 3
        pipe.scheduler.config.use_dynamic_shifting = True

        print("Loading LoRA weights...")
        
        # Load LoRA weights if provided
        if lora_path and os.path.exists(lora_path):
            pipe.load_lora_weights(lora_path)

        print("Moving to device...")
        
        # Apply sequential CPU offloading if requested and device is CPU
        if model_cpu_offload:
            pipe.enable_model_cpu_offload()
        if sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        if not model_cpu_offload and not sequential_cpu_offload:
            pipe.to(device)
            
            


        print("Model loaded successfully")
        
        return (pipe,)


class DSDGeminiPromptEnhancer:
    """Enhances prompts using Google's Gemini API"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False,"tooltip":"Enter your Gemini API key here or use the environment variable GEMINI_API_KEY."})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "enhance_prompt"
    CATEGORY = "DSD"
    OUTPUT_NODE = True  # This ensures that UI data is sent to the node
    
    def __init__(self):
        self.enhanced_prompt = None
    
    def get_state(self):
        return {
            "enhanced_prompt": self.enhanced_prompt
        }
    
    def enhance_prompt(self, image, prompt, api_key):
        if not IMPORTS_AVAILABLE:
            print("Warning: DSD modules not available. Using original prompt.")
            self.enhanced_prompt = None
            return (prompt,)
            
        if not GEMINI_AVAILABLE:
            print("Warning: Google Gemini API not available. Returning original prompt.")
            self.enhanced_prompt = None
            return (prompt,)
        
        if not api_key:
            #try to get api key from environment variable
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("Warning: No API key provided for Gemini. Returning original prompt.")
                self.enhanced_prompt = None
                return (prompt,)
            
        # Convert from ComfyUI image to PIL
        pil_image = comfy_to_pil(image)
        
        # Use the imported enhance_prompt function
        try:
            
            # Call the imported enhance_prompt function
            enhanced_prompt = enhance_prompt(pil_image, prompt, api_key)
            
            print("Original prompt:", prompt)
            print("Enhanced prompt:", enhanced_prompt)
            
            # Store the enhanced prompt for UI display
            self.enhanced_prompt = enhanced_prompt
            
            # Return the enhanced prompt and explicitly include it in the UI data
            # Make sure enhanced_prompt is a proper string, not an array/list of characters
            return {"ui": {"enhanced_prompt": str(enhanced_prompt)}, "result": (enhanced_prompt,)}
        except Exception as e:
            print(f"Error enhancing prompt: {e}")
            self.enhanced_prompt = None
            return (prompt,)


class DSDImageGenerator:
    """Generates images using the DSD model"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dsd_model": ("DSD_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "image_guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "text_guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 512, "max": 2048, "step": 64}),
                "use_gemini_prompt": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "resize_params": ("RESIZE_PARAMS",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("image", "reference_image", "seed")
    FUNCTION = "generate"
    CATEGORY = "DSD"

    def __init__(self):
        self.enhanced_prompt = None
        # For state tracking
        self.progress_value = 0.0
    
    def get_state(self):
        return {
            "progress": self.progress_value,
            "status_text": self.status_text
        }
    
    @property
    def status_text(self):
        if self.enhanced_prompt:
            return f"Enhanced prompt: {self.enhanced_prompt}"
        return ""
    
    def generate(self, dsd_model, image, prompt, negative_prompt, seed,
                guidance_scale, image_guidance_scale, text_guidance_scale, num_inference_steps,
                width, height, use_gemini_prompt, resize_params=None):
        # Initialize progress bar
        pbar = ProgressBar(num_inference_steps)
        # Reset progress value
        self.progress_value = 0.0
        
        # Convert from ComfyUI image format to PIL
        pil_image = comfy_to_pil(image)
        
        # Process the image based on resize_params
        if resize_params is not None:
            method = resize_params.get("method", "center_crop")
            interpolation = resize_params.get("interpolation", "LANCZOS")
            pad_color = resize_params.get("pad_color", (0, 0, 0))
            
            # Apply the selected resize method
            if method == "resize_and_center_crop":
                pil_image = resize_and_center_crop(pil_image, height, width//2)
            elif method == "center_crop":
                pil_image = center_crop(pil_image, height, width//2, interpolation)
            elif method == "pad":
                pil_image = pad_resize(pil_image, height, width//2, pad_color, interpolation)
            elif method == "fit":
                pil_image = fit_resize(pil_image, height, width//2, interpolation)
        else:
            # Use the default center_crop_and_resize if no resize_params provided
            pil_image = resize_and_center_crop(pil_image, height, width//2)
        
        # Clean prompt
        prompt = prompt.strip().replace("\n", "").replace("\r", "")
        negative_prompt = negative_prompt.strip().replace("\n", "").replace("\r", "")

        # If use_gemini_prompt enabled, we've already enhanced the prompt, so just store it to show in the UI
        if use_gemini_prompt:
            try:
                self.enhanced_prompt = prompt
            except Exception as e:
                print(f"Error enhancing prompt: {e}")
                self.enhanced_prompt = None
        else:
            self.enhanced_prompt = None
        
        # Set up progress callback
        def progress_callback(pipe, step, t, callback_kwargs):
            # Update the progress bar
            pbar.update_absolute(step + 1)
            # Update progress value for state tracking
            self.progress_value = (step + 1) / num_inference_steps
            return callback_kwargs
        
        # Set up generator with seed
        if seed == 0:
            # Use a random seed if 0 is provided
            seed = torch.randint(0, 2147483647, (1,)).item()
            print(f"Using random seed: {seed}")
        
        # Create generator with the seed
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        generator.manual_seed(seed)

        
        
        # Run generation
        result = dsd_model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            image=pil_image,
            guidance_scale_real_i=image_guidance_scale,
            guidance_scale_real_t=text_guidance_scale,
            callback_on_step_end=progress_callback,
            generator=generator
        ).images
        
        # Debug information
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result)}")
        
        # Get the output image (right side)
        output_image = None
        if result[0] is not None:
            print(f"Output image type: {type(result[0])}")
            if isinstance(result[0], list):
                print(f"Output image list length: {len(result[0])}")
                if len(result[0]) > 0:
                    print(f"First output image type: {type(result[0][0])}, size: {result[0][0].size if hasattr(result[0][0], 'size') else 'unknown'}")
            else:
                print(f"Output image size: {result[0].size if hasattr(result[0], 'size') else 'unknown'}")
            
            # Convert to ComfyUI format
            output_image = pil_to_comfy(result[0])
            if output_image is not None:
                print(f"Converted output image shape: {output_image.shape}")
        
        # Get the reference image (left side)
        reference_image = None
        if len(result) > 1 and result[1] is not None:
            print(f"Reference image type: {type(result[1])}")
            if isinstance(result[1], list):
                print(f"Reference image list length: {len(result[1])}")
                if len(result[1]) > 0:
                    print(f"First reference image type: {type(result[1][0])}, size: {result[1][0].size if hasattr(result[1][0], 'size') else 'unknown'}")
            else:
                print(f"Reference image size: {result[1].size if hasattr(result[1], 'size') else 'unknown'}")
            
            # Convert to ComfyUI format
            reference_image = pil_to_comfy(result[1])
            if reference_image is not None:
                print(f"Converted reference image shape: {reference_image.shape}")
        
        # Ensure progress is complete
        pbar.update_absolute(num_inference_steps)
        self.progress_value = 1.0
        
      
        
        return (output_image, reference_image, seed)


# Add a separate model selector that helps find models in the model directory
class DSDModelSelector:
    """Selects a DSD model from the models directory"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "check_model_exists": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("model_path", "lora_path")
    FUNCTION = "select_model"
    CATEGORY = "DSD"
    
    def select_model(self, check_model_exists):
        # Get the transformer path
        model_path = os.path.join(dsd_model_path, "transformer", "diffusion_pytorch_model.safetensors")
        # Get the lora path
        lora_path = os.path.join(dsd_model_path, "pytorch_lora_weights.safetensors")
        
        # Check if files exist
        if check_model_exists:
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found at {model_path}")
                print("You can use DSDModelDownloader to download and load the model.")
            
            if not os.path.exists(lora_path):
                print(f"Warning: LoRA file not found at {lora_path}")
                print("You can use DSDModelDownloader to download and load the model.")
        
        return (model_path, lora_path)


class DSDModelDownloader:
    """Downloads and loads the DSD model from Hugging Face"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"default": "primecai/dsd_model"}),
                "HF_Token":("STRING", {"default": "your token here"}),
                "force_download": ("BOOLEAN", {"default": False}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16", "tooltip": "bfloat16 provides best speed/memory tradeoff"}),
                "low_cpu_mem_usage": ("BOOLEAN", {"default": True, "tooltip": "Reduces CPU memory usage during model loading. Recommended for faster loading."}),
                "model_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "Offloads state dict to reduce memory usage during loading. May slow down loading speed."}), 
                "sequential_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "Enables sequential CPU offloading. Only use if low on VRAM. Significantly impacts loading speed."})
            }
        }
    
    RETURN_TYPES = ("DSD_MODEL", "STRING", "STRING")
    RETURN_NAMES = ("dsd_model", "model_path", "lora_path")
    FUNCTION = "download_and_load_model"
    CATEGORY = "DSD"
    
    def __init__(self):
        self.progress_value = 0.0
        self.status_text = ""
    
    def get_state(self):
        return {
            "progress": self.progress_value,
            "status_text": self.status_text
        }
    
    def download_and_load_model(self, repo_id,HF_Token, force_download, device, dtype, low_cpu_mem_usage, model_cpu_offload, sequential_cpu_offload):
        if not IMPORTS_AVAILABLE:
            raise ImportError("Could not import DSD modules. Make sure DSD project files (pipeline.py, transformer.py) are properly installed in the parent directory.")
        # Make sure Hugging Face knows your token
        login(token=HF_Token)                    # ← registers your token
        os.environ["HUGGINGFACE_TOKEN"] = HF_Token  # ← also set env var
        # Create the dsd_model directory in ComfyUI models folder if it doesn't exist
        os.makedirs(dsd_model_path, exist_ok=True)
        transformer_path = os.path.join(dsd_model_path, "transformer")
        os.makedirs(transformer_path, exist_ok=True)
        
        # Check if model already exists
        model_file = os.path.join(transformer_path, "diffusion_pytorch_model.safetensors")
        config_file = os.path.join(transformer_path, "config.json")
        lora_file = os.path.join(dsd_model_path, "pytorch_lora_weights.safetensors")
        
        files_exist = os.path.exists(model_file) and os.path.exists(config_file) and os.path.exists(lora_file)
        
        if not files_exist or force_download:
            self.status_text = f"Downloading DSD model from {repo_id}..."
            print(self.status_text)
            self.progress_value = 0.1
            
            try:
                # Download the model files
                self.status_text = "Downloading transformer model and LoRA weights..."
                print(self.status_text)
                
                # Use snapshot_download to download all files
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=dsd_model_path,
                    local_dir_use_symlinks=False,
                    use_auth_token=HF_Token,
                    resume_download=True
                )
                
                self.progress_value = 0.5
                self.status_text = "Model downloaded successfully"
                print(self.status_text)
                
                # Verify files were downloaded correctly
                if not os.path.exists(model_file):
                    raise FileNotFoundError(f"Model file not found at {model_file} after download. The repository structure may be different than expected.")
                
                if not os.path.exists(config_file):
                    raise FileNotFoundError(f"Config file not found at {config_file} after download. The repository structure may be different than expected.")
                
                if not os.path.exists(lora_file):
                    raise FileNotFoundError(f"LoRA file not found at {lora_file} after download. The repository structure may be different than expected.")
                
            except Exception as e:
                self.status_text = f"Error downloading model: {str(e)}"
                print(self.status_text)
                raise
        else:
            self.status_text = "Model files already exist. Skipping download."
            print(self.status_text)
            self.progress_value = 0.5
        
        # Convert dtype string to torch dtype
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }.get(dtype, torch.bfloat16)
        
        self.status_text = "Loading transformer..."
        print(self.status_text)
        self.progress_value = 0.6
        
        try:
            # Load model with user-specified parameters
            transformer = FluxTransformer2DConditionalModel.from_pretrained(
                transformer_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                ignore_mismatched_sizes=True,
                use_safetensors=True
            )
            
            self.status_text = "Loading pipeline..."
            print(self.status_text)
            self.progress_value = 0.7
            
                        

                        
            # Using black-forest-labs/FLUX.1-schnell,so we don't need to login to Hugging Face
            pipe = FluxConditionalPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                transformer=transformer,
                use_auth_token=HF_Token,
                torch_dtype=torch_dtype
                )
            
            # Access and modify scheduler configs
            pipe.scheduler.config.shift = 3
            pipe.scheduler.config.use_dynamic_shifting = True

            
            
            self.status_text = "Loading LoRA weights..."
            print(self.status_text)
            self.progress_value = 0.8
            
            # Load LoRA weights
            pipe.load_lora_weights(lora_file)
            
            # Apply sequential CPU offloading if requested and device is CPU
            if model_cpu_offload:
                pipe.enable_model_cpu_offload()
            if sequential_cpu_offload:
                pipe.enable_sequential_cpu_offload()
            if not model_cpu_offload and not sequential_cpu_offload:
                pipe.to(device)
            
            self.progress_value = 0.9
            
            self.status_text = "Model loaded successfully"
            print(self.status_text)
            self.progress_value = 1.0
            
            return (pipe, model_file, lora_file)
            
        except Exception as e:
            self.status_text = f"Error loading model: {str(e)}"
            print(self.status_text)
            raise


class DSDResizeSelector:
    """Selects image resize options for DSD Image Generator"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resize_method": (["resize_and_center_crop", "center_crop", "pad", "fit"], {"default": "resize_and_center_crop"}),
                "interpolation": (["LANCZOS", "BICUBIC", "BILINEAR", "NEAREST"], {"default": "LANCZOS"}),
                "pad_r": ("INT", {"default": 0, "min": 0, "max": 255}),
                "pad_g": ("INT", {"default": 0, "min": 0, "max": 255}),
                "pad_b": ("INT", {"default": 0, "min": 0, "max": 255}),
            }
        }
    
    RETURN_TYPES = ("RESIZE_PARAMS",)
    RETURN_NAMES = ("resize_params",)
    FUNCTION = "select_resize_options"
    CATEGORY = "DSD"
    
    def select_resize_options(self, resize_method, interpolation, pad_r, pad_g, pad_b):
        # Create a JSON object with the resize parameters
        resize_params = {
            "method": resize_method,
            "interpolation": interpolation,
            "pad_color": (pad_r, pad_g, pad_b)
        }
        
        return (resize_params,)


# Register nodes
NODE_CLASS_MAPPINGS = {
    "DSDModelLoader": DSDModelLoader,
    "DSDGeminiPromptEnhancer": DSDGeminiPromptEnhancer,
    "DSDImageGenerator": DSDImageGenerator,
    "DSDModelSelector": DSDModelSelector,
    "DSDModelDownloader": DSDModelDownloader,
    "DSDResizeSelector": DSDResizeSelector
}

# Node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "DSDModelLoader": "DSD Model Loader",
    "DSDGeminiPromptEnhancer": "DSD Gemini Prompt Enhancer",
    "DSDImageGenerator": "DSD Image Generator",
    "DSDModelSelector": "DSD Model Selector",
    "DSDModelDownloader": "DSD Model Downloader",
    "DSDResizeSelector": "DSD Resize Selector"
} 
