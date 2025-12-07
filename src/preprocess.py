"""
Image preprocessing utilities for Rice Disease Classification
"""
from torchvision import transforms
from PIL import Image


# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform(image_size=224, augment=False):
    """
    Get image transformation pipeline.
    
    Args:
        image_size (int): Target image size (default: 224)
        augment (bool): Whether to apply data augmentation (default: False)
        
    Returns:
        torchvision.transforms.Compose: Transformation pipeline
    """
    if augment:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    
    return transform


def preprocess_image(image, image_size=224):
    """
    Preprocess an image for model inference.
    
    Args:
        image (PIL.Image): Input image
        image_size (int): Target image size (default: 224)
        
    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 3, image_size, image_size)
    """
    transform = get_transform(image_size=image_size, augment=False)
    
    # Convert to RGB if needed
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    
    # Apply transformations
    tensor = transform(image)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor

