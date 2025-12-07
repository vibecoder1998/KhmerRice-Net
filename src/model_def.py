"""
Neural network model definition for Rice Disease Classification
"""
import torch.nn as nn
from torchvision import models


class RiceCNN(nn.Module):
    """
    CNN model for rice disease classification using transfer learning.
    
    Supports EfficientNet-B0 and ResNet-50 backbones pre-trained on ImageNet.
    """
    def __init__(self, backbone="efficientnet_b0", num_classes=6, freeze_backbone=True):
        """
        Initialize the RiceCNN model.
        
        Args:
            backbone (str): Backbone architecture - "efficientnet_b0" or "resnet50"
            num_classes (int): Number of disease classes (default: 6)
            freeze_backbone (bool): Whether to freeze backbone parameters initially
        """
        super().__init__()

        if backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Identity()

        elif backbone == "resnet50":
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1
            )
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        else:
            raise ValueError("Unsupported backbone. Use efficientnet_b0 or resnet50")

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        feat = self.backbone(x)
        return self.classifier(feat)

