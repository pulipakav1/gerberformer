"""Detection heads for GerberFormer."""
import math
import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        def branch(out_ch):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_ch, 1),
            )

        self.cls_branch = branch(num_anchors * num_classes)
        self.reg_branch = branch(num_anchors * 4)
        self.obj_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, num_anchors, 1),
        )
        self.unc_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, num_anchors, 1),
            nn.Softplus(),
        )
        prior = 0.01
        nn.init.constant_(self.cls_branch[-1].bias, -math.log((1-prior)/prior))
        nn.init.constant_(self.obj_branch[-1].bias, -math.log((1-prior)/prior))

    def forward(self, x):
        B, C, H, W = x.shape
        A, K = self.num_anchors, self.num_classes
        cls = self.cls_branch(x).permute(0,2,3,1).contiguous().view(B,H,W,A,K)
        reg = self.reg_branch(x).permute(0,2,3,1).contiguous().view(B,H,W,A,4)
        obj = self.obj_branch(x).permute(0,2,3,1).contiguous().view(B,H,W,A,1)
        unc = self.unc_branch(x).permute(0,2,3,1).contiguous().view(B,H,W,A,1)
        return cls, reg, obj, unc
