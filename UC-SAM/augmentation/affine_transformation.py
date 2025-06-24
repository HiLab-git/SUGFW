import numpy as np
from scipy import ndimage

def random_rotate(image: np.ndarray, angle: int=None) -> tuple:
    """
    Randomly rotate image and label by same angle
    
    Args:
        image: Input image array
        label: Input label array
        angle_range: Maximum rotation angle in degrees
        
    Returns:
        tuple: (rotated_image, rotated_label)
    """
    if not angle:
        angle = np.random.choice([90, 180, 270])
    
    # Rotate image using bicubic interpolation
    image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False, order=3)
    
    # # Rotate label using nearest neighbor interpolation
    # label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False, order=0)
    
    return image, angle

def random_flip(image: np.ndarray, axis: int=None) -> tuple:
    """
    Randomly flip image and label horizontally or vertically
    
    Args:
        image: Input image array
        label: Input label array
        
    Returns:
        tuple: (flipped_image, flipped_label)
    """
    # Randomly choose flip axis (0=vertical, 1=horizontal)
    if not axis:
        axis = np.random.randint(0, 2)
    
    # Flip both image and label along same axis
    image = np.flip(image, axis=axis).copy()
    # label = np.flip(label, axis=axis).copy()
    
    return image, axis

def random_rot_flip(image: np.ndarray, label: np.ndarray) -> tuple:
    """
    Randomly rotate image by 90 degrees multiples and flip
    
    Args:
        image: Input image array
        label: Input label array
        
    Returns:
        tuple: (transformed_image, transformed_label)
    """
    k = np.random.randint(0, 4)  # 0, 90, 180, or 270 degrees
    
    # Rotate
    image = np.rot90(image, k, axes=(0, 1))
    label = np.rot90(label, k, axes=(0, 1))
    
    # Random flip
    if np.random.random() > 0.5:
        image, label = random_flip(image, label)
        
    return image, label