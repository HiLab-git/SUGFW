import random
import numpy as np
from PIL import Image, ImageEnhance

def gamma_correction(image: np.ndarray) -> np.ndarray:
    """
    Apply gamma correction to an image.
    
    Args:
        image (np.ndarray): Input image array (values should be normalized to [0, 1]).
        gamma (float): Gamma value for the transformation.
        
    Returnsa:
        np.ndarray: Gamma-corrected image.
    """
    # Ensure the input image is in the range [0, 1]
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Input image values should be normalized to the range [0, 1].")
    
    gamma = random.choice([0.1, 0.5, 1.5, 1.9])
    # Apply gamma correction
    corrected_image = np.power(image, gamma)
    return corrected_image, gamma

def posterization(image: np.ndarray) -> np.ndarray:
    """
    Apply posterization to an image by reducing the number of bits per channel.
    
    Args:
        image (np.ndarray): Input image array (values should be normalized to [0, 1]).
        bits (int): Number of bits to keep (e.g., 4 for 16 levels).
        
    Returns:
        np.ndarray: Posterized image.
    """
    bits = random.randint(4, 9)
    levels = 2 ** bits
    posterized_image = np.floor(image * levels) / levels
    return posterized_image, bits

def contrast_adjustment(image: np.ndarray) -> np.ndarray:
    """
    Adjust the contrast of an image.
    
    Args:
        image (np.ndarray): Input image array (values should be normalized to [0, 1]).
        factor (float): Contrast adjustment factor (1.0 = no change, >1.0 = increase, <1.0 = decrease).
        
    Returns:
        np.ndarray: Contrast-adjusted image.
    """
    factor = random.choice([0.1, 0.5, 1.5, 1.9])
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    enhancer = ImageEnhance.Contrast(pil_image)
    adjusted_image = enhancer.enhance(factor)
    return np.array(adjusted_image) / 255.0, factor

def brightness_modification(image: np.ndarray) -> np.ndarray:
    """
    Modify the brightness of an image.
    
    Args:
        image (np.ndarray): Input image array (values should be normalized to [0, 1]).
        factor (float): Brightness adjustment factor (1.0 = no change, >1.0 = increase, <1.0 = decrease).
        
    Returns:
        np.ndarray: Brightness-modified image.
    """
    factor = random.choice([0.1, 0.5, 1.5, 1.9])
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    enhancer = ImageEnhance.Brightness(pil_image)
    adjusted_image = enhancer.enhance(factor)
    return np.array(adjusted_image) / 255.0, factor

def sharpness_enhancement(image: np.ndarray) -> np.ndarray:
    """
    Enhance the sharpness of an image.
    
    Args:
        image (np.ndarray): Input image array (values should be normalized to [0, 1]).
        factor (float): Sharpness adjustment factor (1.0 = no change, >1.0 = increase, <1.0 = decrease).
        
    Returns:
        np.ndarray: Sharpness-enhanced image.
    """
    factor = random.choice([0.1, 0.5, 1.5, 1.9])
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    enhancer = ImageEnhance.Sharpness(pil_image)
    enhanced_image = enhancer.enhance(factor)
    return np.array(enhanced_image) / 255.0, factor

if __name__ == "__main__":
    # Example usage
    # Create a sample normalized image
    sample_image = np.random.rand(256, 256)  # Random image with values in [0, 1]
    
    # Apply gamma correction
    gamma_corrected = gamma_correction(sample_image, gamma=1.5)
    print("Applied gamma correction.")
    
    # Apply posterization
    posterized = posterization(sample_image, bits=4)
    print("Applied posterization.")
    
    # Adjust contrast
    contrast_adjusted = contrast_adjustment(sample_image, factor=1.2)
    print("Adjusted contrast.")
    
    # Modify brightness
    brightness_modified = brightness_modification(sample_image, factor=0.8)
    print("Modified brightness.")
    
    # Enhance sharpness
    sharpness_enhanced = sharpness_enhancement(sample_image, factor=2.0)
    print("Enhanced sharpness.")