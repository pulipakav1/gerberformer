"""Gerber prior encoder."""
import torch.nn as nn

class GerberEncoder(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,  32,  3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64,  3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, gerber):
        return self.proj(self.encoder(gerber))
