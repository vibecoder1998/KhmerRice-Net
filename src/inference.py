"""
Model loading and prediction logic for Rice Disease Classification
"""
from pathlib import Path
import torch
from PIL import Image

from src.model_def import RiceCNN
from src.preprocess import preprocess_image


def load_model_and_metadata(checkpoint_path=None):
    """
    Load the trained model and metadata from checkpoint.
    
    Args:
        checkpoint_path (str or Path): Path to model checkpoint. 
                                      Defaults to "src/models/rice_cnn_model.pth"
        
    Returns:
        tuple: (model, classes, transform_info) where:
            - model: Loaded RiceCNN model in eval mode
            - classes: List of class names
            - transform_info: Dictionary with transform parameters
    """
    if checkpoint_path is None:
        # Try multiple possible locations
        possible_paths = [
            Path("src/models/rice_cnn_model.pth"),
        ]
        checkpoint_path = None
        for path in possible_paths:
            if path.exists():
                checkpoint_path = path
                break
        if checkpoint_path is None:
            checkpoint_path = possible_paths[0]  # Default to models/ directory
    else:
        checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    classes = ckpt["classes"]
    backbone = ckpt.get("backbone", "efficientnet_b0")

    # Initialize model
    model = RiceCNN(backbone=backbone, num_classes=len(classes), freeze_backbone=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Store transform info for reference
    transform_info = {
        "image_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

    return model, classes, transform_info


def predict(model, image, classes=None, top_k=3):
    """
    Make prediction on an image.
    
    Args:
        model (RiceCNN): Loaded model
        image (PIL.Image or torch.Tensor): Input image
        classes (list): List of class names. If None, returns indices
        top_k (int): Number of top predictions to return (default: 3)
        
    Returns:
        tuple: (top_label, confidence, all_probs) where:
            - top_label: Predicted class name or index
            - confidence: Confidence score (0-1)
            - all_probs: List of (label, probability) tuples sorted by probability
    """
    # Preprocess image if it's a PIL Image
    if isinstance(image, Image.Image):
        img_tensor = preprocess_image(image)
    else:
        img_tensor = image
    
    # Ensure model is in eval mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    
    # Get top predictions
    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))
    
    # Format results
    all_probs = []
    for idx, prob in enumerate(probs):
        label = classes[idx] if classes else idx
        all_probs.append((label, float(prob)))
    
    # Sort by probability
    all_probs.sort(key=lambda x: x[1], reverse=True)
    
    # Get top prediction
    top_label = all_probs[0][0]
    confidence = all_probs[0][1]
    
    return top_label, confidence, all_probs

