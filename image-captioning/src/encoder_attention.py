from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import ResNet101_Weights, resnet101


class EncoderAttentionCNN(nn.Module):
    """CNN encoder that outputs projected spatial features for attention-based decoders.

    Input shape: (B, 3, 224, 224)
    Output shape: (B, 49, encoder_dim)
    """

    def __init__(self, encoder_dim: int = 512) -> None:
        super().__init__()

        # Keep ResNet-101 up to layer4 (exclude avgpool and fc).
        backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)

        # Freeze all ResNet parameters before truncating the model.
        for param in backbone.parameters():
            param.requires_grad = False

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # Project each spatial feature vector (2048 -> encoder_dim).
        self.projection = nn.Linear(2048, encoder_dim)

    def forward(self, images: Tensor) -> Tensor:
        """Encode images into projected spatial features.

        Args:
            images: Tensor of shape (B, 3, 224, 224).

        Returns:
            Tensor of shape (B, 49, encoder_dim).
        """
        features = self.backbone(images)  # (B, 2048, 7, 7) for 224x224 inputs
        batch_size, channels, height, width = features.shape

        if channels != 2048:
            raise ValueError(
                f"Expected 2048 channels from ResNet layer4, got {channels}.")

        # Rearrange spatial grid to sequence of locations: (B, H*W, C) = (B, 49, 2048).
        spatial_features = features.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channels)

        # Project each location embedding to encoder_dim.
        projected_features = self.projection(
            spatial_features)  # (B, 49, encoder_dim)
        return projected_features
