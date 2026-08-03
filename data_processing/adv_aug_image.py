"""
Advanced image augmentation for chest X-rays.
Implements more sophisticated augmentation techniques while preserving medical validity.
"""

import numpy as np
from scipy import ndimage
import random

def apply_elastic_deformation(image, alpha=50, sigma=5, random_state=None):
    """
    Apply elastic deformation to simulate patient positioning variance.
    
    Args:
        image: Input image (height, width, channels)
        alpha: Deformation intensity
        sigma: Deformation smoothness
        random_state: Random seed for reproducibility
        
    Returns:
        Deformed image same shape as input
    """
    random_state = np.random.RandomState(random_state)
    
    shape = image.shape[:2]
    
    # Random displacement fields
    dx = random_state.rand(*shape) * 2 - 1
    dy = random_state.rand(*shape) * 2 - 1
    
    # Smooth displacement fields
    dx = ndimage.gaussian_filter(dx, sigma, mode="constant", cval=0) * alpha
    dy = ndimage.gaussian_filter(dy, sigma, mode="constant", cval=0) * alpha
    
    # Create meshgrid for displacement
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    
    # Apply displacement field to all channels
    distorted_image = np.zeros_like(image)
    for c in range(image.shape[2]):
        distorted_image[:,:,c] = ndimage.map_coordinates(image[:,:,c], indices, order=1).reshape(shape)
    
    return distorted_image

def apply_advanced_image_augmentation(image, config):
    """
    Apply advanced augmentations to a chest X-ray image
    
    Args:
        image: Input image as numpy array (h, w, c)
        config: AdvAugConfig object with augmentation parameters
        
    Returns:
        Augmented image
    """
    # Make a copy to avoid modifying original
    img = image.copy()
    
    # Translation (shift) - more aggressive
    if random.random() < 0.7:
        shift_x = random.uniform(-config.translation_range, config.translation_range)
        shift_y = random.uniform(-config.translation_range, config.translation_range)
        img = ndimage.shift(img, (shift_y * img.shape[0], shift_x * img.shape[1], 0), mode='nearest')
    
    # Rotation - increased range
    if random.random() < 0.7:
        angle = random.uniform(-config.rotation_range, config.rotation_range)
        img = ndimage.rotate(img, angle, reshape=False, mode='nearest')
    
    # Brightness adjustment - more variation
    if random.random() < 0.7:
        brightness_factor = random.uniform(1-config.brightness_range, 1+config.brightness_range)
        img = np.clip(img * brightness_factor, 0, 1)
    
    # Contrast adjustment - more aggressive
    if random.random() < 0.6:
        contrast_factor = random.uniform(1-config.contrast_range, 1+config.contrast_range)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = np.clip((img - mean) * contrast_factor + mean, 0, 1)
    
    # Add Gaussian noise - slightly increased
    if random.random() < 0.5:
        noise = np.random.normal(0, config.noise_level, img.shape)
        img = np.clip(img + noise, 0, 1)
    
    # Zoom (scale) - more variation
    if random.random() < 0.6:
        zoom_factor = random.uniform(1-config.zoom_range, 1+config.zoom_range)
        
        try:
            img = ndimage.zoom(img, (zoom_factor, zoom_factor, 1), mode='nearest', order=1)
            
            # Handle size change (crop or pad to original dimensions)
            h, w = image.shape[:2]
            
            if img.shape[0] > h:  # Need to crop
                start_h = (img.shape[0] - h) // 2
                start_w = (img.shape[1] - w) // 2
                img = img[start_h:start_h+h, start_w:start_w+w]
            elif img.shape[0] < h:  # Need to pad
                padded_img = np.zeros_like(image)
                start_h = (h - img.shape[0]) // 2
                start_w = (w - img.shape[1]) // 2
                end_h = min(start_h + img.shape[0], h)
                end_w = min(start_w + img.shape[1], w)
                h_to_copy = end_h - start_h
                w_to_copy = end_w - start_w
                
                padded_img[start_h:end_h, start_w:end_w] = img[:h_to_copy, :w_to_copy]
                img = padded_img
        except Exception as e:
            print(f"Zoom error handled: {e}")
            img = image.copy()
    
    # Apply elastic deformation - new augmentation
    if config.use_elastic_deform and random.random() < 0.5:
        try:
            img = apply_elastic_deformation(
                img, 
                alpha=config.elastic_deform_alpha,
                sigma=config.elastic_deform_sigma,
                random_state=random.randint(0, 1000)
            )
        except Exception as e:
            print(f"Elastic deformation error handled: {e}")
    
    return img
