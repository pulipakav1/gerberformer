"""Feature Pyramid Network neck."""
import torch.nn as nn
import torch.nn.functional as F

class FPNNeck(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            ) for _ in in_channels])

    def forward(self, features):
        laterals = [l(f) for l, f in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], mode="nearest")
        return [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
