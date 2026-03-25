import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

NUM_CLASSES = 6
NUM_ANCHORS = 3

ANCHORS = [
    [(0.02, 0.02), (0.04, 0.04), (0.08, 0.08)],
    [(0.12, 0.12), (0.18, 0.18), (0.25, 0.25)],
    [(0.35, 0.35), (0.50, 0.50), (0.70, 0.70)],
]


class FPNNeck(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ) for _ in in_channels
        ])

    def forward(self, features):
        laterals = [l(f) for l, f in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], mode="nearest")
        return [conv(lat) for conv, lat in zip(self.output_convs, laterals)]


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
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_anchors, 1),
        )
        self.unc_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_anchors, 1),
            nn.Softplus(),
        )

        prior = 0.01
        nn.init.constant_(self.cls_branch[-1].bias, -math.log((1 - prior) / prior))
        nn.init.constant_(self.obj_branch[-1].bias, -math.log((1 - prior) / prior))

    def forward(self, x):
        B, C, H, W = x.shape
        A, K = self.num_anchors, self.num_classes
        cls = self.cls_branch(x).permute(0,2,3,1).contiguous().view(B,H,W,A,K)
        reg = self.reg_branch(x).permute(0,2,3,1).contiguous().view(B,H,W,A,4)
        obj = self.obj_branch(x).permute(0,2,3,1).contiguous().view(B,H,W,A,1)
        unc = self.unc_branch(x).permute(0,2,3,1).contiguous().view(B,H,W,A,1)
        return cls, reg, obj, unc


class GerberFormer(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS,
                 fpn_channels=256, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 3),
            img_size=640,
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 640, 640)
            feats = self.backbone(dummy)
            backbone_channels = [f.shape[1] for f in feats]
        self.neck  = FPNNeck(backbone_channels, out_channels=fpn_channels)
        self.heads = nn.ModuleList([
            DetectionHead(fpn_channels, num_classes, num_anchors)
            for _ in backbone_channels
        ])
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[GerberFormer] {n:,} parameters | {len(backbone_channels)} scales")

    def forward(self, images, design_features=None):
        feats = self.backbone(images)
        feats = self.neck(feats)
        outputs = []
        for f, head in zip(feats, self.heads):
            cls, reg, obj, unc = head(f)
            outputs.append({"cls": cls, "reg": reg, "obj": obj, "unc": unc})
        return outputs


def build_targets(predictions, targets, anchors):
    device = predictions[0]["obj"].device
    scale_targets = []
    for scale_idx, pred in enumerate(predictions):
        B, H, W, A, _ = pred["obj"].shape
        K = pred["cls"].shape[-1]
        obj_target = torch.zeros(B, H, W, A,    device=device)
        cls_target = torch.zeros(B, H, W, A, K, device=device)
        reg_target = torch.zeros(B, H, W, A, 4, device=device)
        obj_mask   = torch.zeros(B, H, W, A,    device=device, dtype=torch.bool)
        noobj_mask = torch.ones( B, H, W, A,    device=device, dtype=torch.bool)
        anc = torch.tensor(anchors[scale_idx], device=device)
        for b in range(B):
            gt    = targets[b]
            valid = gt[:, 0] >= 0
            gt    = gt[valid]
            if len(gt) == 0:
                continue
            for box in gt:
                cls_id      = box[0].long()
                cx, cy, bw, bh = box[1], box[2], box[3], box[4]
                gt_wh  = torch.stack([bw, bh]).unsqueeze(0).expand(A, -1)
                inter  = torch.min(gt_wh, anc).prod(dim=1)
                union  = gt_wh.prod(dim=1) + anc.prod(dim=1) - inter
                ious   = inter / (union + 1e-7)
                best_a = ious.argmax()
                gi = min(int(cx * W), W - 1)
                gj = min(int(cy * H), H - 1)
                obj_target[b, gj, gi, best_a]         = 1.0
                obj_mask  [b, gj, gi, best_a]         = True
                noobj_mask[b, gj, gi, best_a]         = False
                cls_target[b, gj, gi, best_a, cls_id] = 1.0
                reg_target[b, gj, gi, best_a]         = torch.stack([cx, cy, bw, bh])
        scale_targets.append({
            "obj": obj_target, "cls": cls_target,
            "reg": reg_target, "obj_mask": obj_mask,
            "noobj_mask": noobj_mask,
        })
    return scale_targets


class GerberFormerLoss(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_reg=2.0,
                 lambda_obj=1.0, lambda_noobj=0.5, lambda_unc=0.05,
                 focal_gamma=2.0, focal_alpha=0.25):
        super().__init__()
        self.lambda_cls   = lambda_cls
        self.lambda_reg   = lambda_reg
        self.lambda_obj   = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_unc   = lambda_unc
        self.focal_gamma  = focal_gamma
        self.focal_alpha  = focal_alpha

    def focal_loss(self, pred, target):
        p   = torch.sigmoid(pred)
        ce  = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        a_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        return (a_t * (1 - p_t) ** self.focal_gamma * ce).mean()

    def forward(self, predictions, targets):
        device = predictions[0]["obj"].device
        scale_targets = build_targets(predictions, targets, ANCHORS)
        total      = torch.tensor(0.0, device=device)
        components = {"loss_obj": 0., "loss_cls": 0.,
                      "loss_reg": 0., "loss_unc": 0.}
        for pred, tgt in zip(predictions, scale_targets):
            obj_mask   = tgt["obj_mask"]
            noobj_mask = tgt["noobj_mask"]
            if obj_mask.sum() > 0:
                l_obj = F.binary_cross_entropy_with_logits(
                    pred["obj"].squeeze(-1)[obj_mask], tgt["obj"][obj_mask])
                l_cls = self.focal_loss(pred["cls"][obj_mask], tgt["cls"][obj_mask])
                l_reg = F.smooth_l1_loss(pred["reg"][obj_mask], tgt["reg"][obj_mask])
            else:
                l_obj = l_cls = l_reg = torch.tensor(0., device=device)
            l_noobj = F.binary_cross_entropy_with_logits(
                pred["obj"].squeeze(-1)[noobj_mask], tgt["obj"][noobj_mask])
            l_unc   = pred["unc"].mean()
            scale_loss = (self.lambda_obj   * l_obj   +
                          self.lambda_noobj * l_noobj +
                          self.lambda_cls   * l_cls   +
                          self.lambda_reg   * l_reg   +
                          self.lambda_unc   * l_unc)
            total = total + scale_loss
            components["loss_obj"] += l_obj.item()
            components["loss_cls"] += l_cls.item()
            components["loss_reg"] += l_reg.item()
            components["loss_unc"] += l_unc.item()
        components["loss"] = total
        return components
