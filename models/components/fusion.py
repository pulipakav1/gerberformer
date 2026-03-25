"""Cross-modal fusion module."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalFusion(nn.Module):
    """
    Memory-efficient cross-modal fusion via depthwise-separable convolution.
    Fuses image FPN features with Gerber design features.
    """
    def __init__(self, channels=256):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels*2, channels*2, 3, padding=1, groups=channels*2),
            nn.Conv2d(channels*2, channels, 1),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels*2, channels), nn.Sigmoid(),
        )
        self.norm = nn.GroupNorm(32, channels)

    def forward(self, image_feat, gerber_feat):
        B, C, H, W = image_feat.shape
        gerber_resized = F.interpolate(gerber_feat, size=(H,W),
                                        mode="bilinear", align_corners=False)
        combined = torch.cat([image_feat, gerber_resized], dim=1)
        fused    = self.fuse(combined)
        gate     = self.gate(combined).view(B, C, 1, 1)
        return self.norm(image_feat + gate * fused)
