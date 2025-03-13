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

def resize_and_center_crop(image: Union[torch.Tensor, Image.Image], target_height: int = 512,target_width: int = 512) -> Union[torch.Tensor, Image.Image]:
    """
    Center crop and resize an image.
    
    Args:
        image: Image to process (ComfyUI tensor or PIL Image)
        target_height: Target height
        target_width: Target width
        
    Returns:
        Processed image in the same format as input
    """
    # Handle ComfyUI tensor
    if isinstance(image, torch.Tensor):
        pil_image = comfy_to_pil(image)
        result = resize_and_center_crop(pil_image, target_height, target_width)
        return pil_to_comfy(result)
    
    # Handle PIL Image
    w, h = image.size
    min_size = min(w, h)

    # Calculate target aspect ratio
    target_ratio = target_width / target_height
    # Calculate current aspect ratio
    current_ratio = w / h

    # Resize to match target width or height while preserving aspect ratio
    if current_ratio > target_ratio:
        # Image is wider than target - resize by height
        new_height = target_height
        new_width = int(w * (target_height / h))
    else:
        # Image is taller than target - resize by width
        new_width = target_width
        new_height = int(h * (target_width / w))

    image = image.resize((new_width, new_height), Image.BILINEAR)

    # Center crop the image to the target size
    cropped = image.crop(((new_width - target_width) // 2, 
                         (new_height - target_height) // 2, 
                         (new_width + target_width) // 2, 
                         (new_height + target_height) // 2))

    return cropped

def center_crop(image: Union[torch.Tensor, Image.Image], target_height: int = 512, target_width: int = 512, 
                interpolation: str = "LANCZOS") -> Union[torch.Tensor, Image.Image]:
    """
    Center crop an image to target size. If image is smaller than target size,
    resize first before cropping.
    
    Args:
        image: Image to process (ComfyUI tensor or PIL Image)
        target_height: Target height
        target_width: Target width
        interpolation: Interpolation method (NEAREST, BILINEAR, BICUBIC, LANCZOS)
        
    Returns:
        Processed image in the same format as input
    """
    # Handle ComfyUI tensor
    if isinstance(image, torch.Tensor):
        pil_image = comfy_to_pil(image)
        result = center_crop(pil_image, target_height, target_width, interpolation)
        return pil_to_comfy(result)
    
    # Handle PIL Image
    w, h = image.size
    
    # Get interpolation method
    interp_method = {
        "NEAREST": Image.NEAREST,
        "BILINEAR": Image.BILINEAR,
        "BICUBIC": Image.BICUBIC,
        "LANCZOS": Image.LANCZOS
    }.get(interpolation, Image.LANCZOS)

    # If image is smaller than target in either dimension, resize first
    if w < target_width or h < target_height:
        # Calculate scale factor needed
        scale = max(target_width / w, target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), interp_method)
        w, h = new_w, new_h

    # Calculate crop coordinates
    left = (w - target_width) // 2
    top = (h - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    # Perform center crop
    cropped = image.crop((left, top, right, bottom))
    return cropped

def pad_resize(image: Union[torch.Tensor, Image.Image], target_height: int = 512, target_width: int = 512, 
               pad_color: tuple = (0, 0, 0), interpolation: str = "LANCZOS") -> Union[torch.Tensor, Image.Image]:
    """
    Resize image preserving aspect ratio and pad to target size.
    
    Args:
        image: Image to process (ComfyUI tensor or PIL Image)
        target_height: Target height
        target_width: Target width
        pad_color: RGB color tuple for padding (default: black)
        interpolation: Interpolation method (NEAREST, BILINEAR, BICUBIC, LANCZOS)
        
    Returns:
        Processed image in the same format as input
    """
    # Handle ComfyUI tensor
    if isinstance(image, torch.Tensor):
        pil_image = comfy_to_pil(image)
        result = pad_resize(pil_image, target_height, target_width, pad_color, interpolation)
        return pil_to_comfy(result)
    
    # Handle PIL Image
    w, h = image.size
    
    # Get interpolation method
    interp_method = {
        "NEAREST": Image.NEAREST,
        "BILINEAR": Image.BILINEAR,
        "BICUBIC": Image.BICUBIC,
        "LANCZOS": Image.LANCZOS
    }.get(interpolation, Image.LANCZOS)
    
    # Calculate target aspect ratio
    target_ratio = target_width / target_height
    # Calculate current aspect ratio
    current_ratio = w / h
    
    # Create a new image with the target size and fill with the pad color
    new_image = Image.new("RGB", (target_width, target_height), pad_color)
    
    # Resize the original image preserving aspect ratio
    if current_ratio > target_ratio:
        # Image is wider than target - resize by width
        new_w = target_width
        new_h = int(h * (target_width / w))
        resized = image.resize((new_w, new_h), interp_method)
        # Paste in the center (horizontally)
        new_image.paste(resized, (0, (target_height - new_h) // 2))
    else:
        # Image is taller than target - resize by height
        new_h = target_height
        new_w = int(w * (target_height / h))
        resized = image.resize((new_w, new_h), interp_method)
        # Paste in the center (vertically)
        new_image.paste(resized, ((target_width - new_w) // 2, 0))
    
    return new_image

def fit_resize(image: Union[torch.Tensor, Image.Image], target_height: int = 512, target_width: int = 512,
               interpolation: str = "LANCZOS") -> Union[torch.Tensor, Image.Image]:
    """
    Resize image to target size without preserving aspect ratio.
    
    Args:
        image: Image to process (ComfyUI tensor or PIL Image)
        target_height: Target height
        target_width: Target width
        interpolation: Interpolation method (NEAREST, BILINEAR, BICUBIC, LANCZOS)
        
    Returns:
        Processed image in the same format as input
    """
    # Handle ComfyUI tensor
    if isinstance(image, torch.Tensor):
        pil_image = comfy_to_pil(image)
        result = fit_resize(pil_image, target_height, target_width, interpolation)
        return pil_to_comfy(result)
    
    # Handle PIL Image
    # Get interpolation method
    interp_method = {
        "NEAREST": Image.NEAREST,
        "BILINEAR": Image.BILINEAR,
        "BICUBIC": Image.BICUBIC,
        "LANCZOS": Image.LANCZOS
    }.get(interpolation, Image.LANCZOS)
    
    # Resize directly to target size (no aspect ratio preservation)
    resized = image.resize((target_width, target_height), interp_method)
    return resized