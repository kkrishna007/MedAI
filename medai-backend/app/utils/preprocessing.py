from PIL import Image
import numpy as np
import io
import cv2

def preprocess_image(image_bytes, target_size=(224, 224)):
    """
    Preprocess image bytes for model prediction
    
    Args:
        image_bytes: Raw image bytes
        target_size: Target size for resizing (default: 224x224)
        
    Returns:
        Preprocessed image as numpy array with batch dimension
    """
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize(target_size)
    image = image.convert("RGB")
    image_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

def get_original_image(image_bytes, target_size=(224, 224)):
    """
    Get original image without normalization for visualization
    
    Args:
        image_bytes: Raw image bytes
        target_size: Target size for resizing (default: 224x224)
        
    Returns:
        Original image as numpy array without batch dimension
    """
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize(target_size)
    image = image.convert("RGB")
    return np.array(image)

def preprocess_brain_image(image_bytes, target_size=(150, 150)):
    """
    Preprocess brain MRI images using OpenCV
    
    Args:
        image_bytes: Raw image bytes
        target_size: Target size for resizing (default: 150x150 for brain tumor model)
        
    Returns:
        Preprocessed image as numpy array with batch dimension
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Resize
    img = cv2.resize(img, target_size)
    # Reshape to add batch dimension
    img = np.reshape(img, (1, target_size[0], target_size[1], 3))
    return img

def preprocess_batch(images, target_size=(224, 224)):
    """
    Preprocess a batch of images for XAI techniques
    
    Args:
        images: Numpy array of images
        target_size: Target size for resizing
        
    Returns:
        Preprocessed batch of images
    """
    if len(images.shape) == 3:  # Single image
        images = np.expand_dims(images, axis=0)
        
    # Ensure images are float32 and normalized
    if images.dtype != np.float32:
        images = images.astype(np.float32)
        
    if images.max() > 1.0:
        images = images / 255.0
        
    return images




# from PIL import Image
# import numpy as np
# import io

# def preprocess_image(image_bytes, target_size=(224, 224)):
#     """
#     Preprocess image bytes for model prediction
    
#     Args:
#         image_bytes: Raw image bytes
#         target_size: Target size for resizing (default: 224x224)
        
#     Returns:
#         Preprocessed image as numpy array with batch dimension
#     """
#     image = Image.open(io.BytesIO(image_bytes))
#     image = image.resize(target_size)
#     image = image.convert("RGB")
#     image_array = np.array(image) / 255.0  # Normalize
#     return np.expand_dims(image_array, axis=0)  # Add batch dimension
