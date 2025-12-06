"""
Source code modules for Rice Disease Classification
"""
from src.model_def import RiceCNN
from src.preprocess import preprocess_image, get_transform
from src.inference import load_model_and_metadata, predict
from src.rice_dataset import RiceDiseaseDataset

# Backward compatibility aliases
from src.model_def import RiceCNN as RiceModel

__all__ = [
    "RiceCNN",
    "RiceModel",  # Backward compatibility
    "preprocess_image",
    "get_transform",
    "load_model_and_metadata",
    "predict",
    "RiceDiseaseDataset",
]

