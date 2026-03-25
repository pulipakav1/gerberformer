"""
Loss functions for GerberFormer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ANCHORS = [
    [(0.02, 0.02), (0.04, 0.04), (0.08, 0.08)],
    [(0.12, 0.12), (0.18, 0.18), (0.25, 0.25)],
    [(0.35, 0.35), (0.50, 0.50), (0.70, 0.70)],
]

def build_targets(predictions, targets, anchors=ANCHORS):
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
                cls_id = box[0].long()
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
            "obj": obj_target, "cls": cls_target, "reg": reg_target,
            "obj_mask": obj_mask, "noobj_mask": noobj_mask,
        })
    return scale_targets


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        p   = torch.sigmoid(pred)
        ce  = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        a_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (a_t * (1 - p_t) ** self.gamma * ce).mean()


class GerberFormerLoss(nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_reg=2.0, lambda_obj=1.0,
                 lambda_noobj=0.5, lambda_unc=0.05):
        super().__init__()
        self.lambda_cls   = lambda_cls
        self.lambda_reg   = lambda_reg
        self.lambda_obj   = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_unc   = lambda_unc
        self.focal        = FocalLoss()

    def forward(self, predictions, targets):
        device = predictions[0]["obj"].device
        tgts   = build_targets(predictions, targets)
        total  = torch.tensor(0.0, device=device)
        logs   = {"loss_obj":0., "loss_cls":0., "loss_reg":0., "loss_unc":0.}
        for pred, tgt in zip(predictions, tgts):
            om, nm = tgt["obj_mask"], tgt["noobj_mask"]
            if om.sum() > 0:
                l_obj = F.binary_cross_entropy_with_logits(
                    pred["obj"].squeeze(-1)[om], tgt["obj"][om])
                l_cls = self.focal(pred["cls"][om], tgt["cls"][om])
                l_reg = F.smooth_l1_loss(pred["reg"][om], tgt["reg"][om])
            else:
                l_obj = l_cls = l_reg = torch.tensor(0., device=device)
            l_noobj = F.binary_cross_entropy_with_logits(
                pred["obj"].squeeze(-1)[nm], tgt["obj"][nm])
            l_unc = pred["unc"].mean()
            total = total + (self.lambda_obj*l_obj + self.lambda_noobj*l_noobj +
                             self.lambda_cls*l_cls + self.lambda_reg*l_reg +
                             self.lambda_unc*l_unc)
            logs["loss_obj"] += l_obj.item()
            logs["loss_cls"] += l_cls.item()
            logs["loss_reg"] += l_reg.item()
            logs["loss_unc"] += l_unc.item()
        logs["loss"] = total
        return logs
