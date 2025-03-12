import os
import torch
import numpy as np
from PIL import Image
from typing import Union, List, Optional

def get_model_path(model_name: str) -> str:
    """
    Get the path to a model file in the models directory.
    
    Args:
        model_name: Name of the model file or directory
        
    Returns:
        Full path to the model
    """
    # Determine the path to the models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    
    # Check if the model exists
    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_name} not found in {models_dir}")
    
    return model_path

def get_lora_path(lora_name: str) -> str:
    """
    Get the path to a LoRA file in the loras directory.
    
    Args:
        lora_name: Name of the LoRA file
        
    Returns:
        Full path to the LoRA file
    """
    # Determine the path to the loras directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    loras_dir = os.path.join(current_dir, "loras")
    
    # Check if the LoRA file exists
    lora_path = os.path.join(loras_dir, lora_name)
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA file {lora_name} not found in {loras_dir}")
    
    return lora_path

def comfy_to_pil(image: torch.Tensor) -> Image.Image:
    """
    Convert a ComfyUI image tensor to a PIL Image.
    
    Args:
        image: ComfyUI image tensor (1, H, W, 3) in range [0, 1]
        
    Returns:
        PIL Image
    """
    # Convert to numpy array and scale to [0, 255]
    image_np = np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8)
    # Convert to PIL Image
    return Image.fromarray(image_np)

def pil_to_comfy(image: Union[Image.Image, List[Image.Image], None]) -> Optional[torch.Tensor]:
    """
    Convert a PIL Image or list of PIL Images to a ComfyUI image tensor.
    
    Args:
        image: PIL Image, list of PIL Images, or None
        
    Returns:
        ComfyUI image tensor (1, H, W, 3) in range [0, 1] or None if input is None
    """
    if image is None:
        return None
    
    # Handle list of PIL images - take the first one
    if isinstance(image, list):
        if len(image) == 0:
            return None
        image = image[0]  # Take the first image from the list
    
    # Convert to numpy array and scale to [0, 1]
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # Ensure the image has the right shape (H, W, 3)
    if len(image_np.shape) == 2:  # Grayscale image
        image_np = np.stack([image_np, image_np, image_np], axis=-1)
    elif image_np.shape[-1] == 4:  # RGBA image
        image_np = image_np[..., :3]  # Remove alpha channel
    
    # Convert to torch tensor and add batch dimension
    return torch.from_numpy(image_np)[None,]

def center_crop_and_resize(image: Union[torch.Tensor, Image.Image], target_size: int = 512) -> Union[torch.Tensor, Image.Image]:
    """
    Center crop and resize an image.
    
    Args:
        image: Image to process (ComfyUI tensor or PIL Image)
        target_size: Target size for width and height
        
    Returns:
        Processed image in the same format as input
    """
    # Handle ComfyUI tensor
    if isinstance(image, torch.Tensor):
        pil_image = comfy_to_pil(image)
        result = center_crop_and_resize(pil_image, target_size)
        return pil_to_comfy(result)
    
    # Handle PIL Image
    w, h = image.size
    min_size = min(w, h)
    cropped = image.crop(((w - min_size) // 2, 
                         (h - min_size) // 2, 
                         (w + min_size) // 2, 
                         (h + min_size) // 2))
    resized = cropped.resize((target_size, target_size), Image.LANCZOS)
    return resized 