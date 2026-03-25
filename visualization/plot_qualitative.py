"""
Generate Figure 3: qualitative detection results on unseen boards.
Usage: python plot_qualitative.py
       --baseline  path/to/baseline/best.pt
       --gerber    path/to/gerber/best.pt
       --data_root pcb-defect-dataset
       --fused_dir pcb-fused-dataset
       --output    paper/figures/fig3_qualitative.pdf
"""
import argparse, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from ultralytics import YOLO

CLASS_NAMES = ["missing_hole","mouse_bite","open_circuit",
               "short","spurious_copper","spur"]
COLORS = ["#FF4444","#FF8800","#FFFF00","#44FF44","#4444FF","#FF44FF"]
UNSEEN = ["light_05","light_07","light_11","light_12"]

def get_base_stem(filename):
    return Path(filename).stem.rsplit("_", 1)[0]

def draw_boxes(ax, img, boxes, title):
    ax.imshow(img)
    h, w = img.shape[:2]
    for cls, cx, cy, bw, bh in boxes:
        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        rect = patches.Rectangle(
            (x1, y1), bw*w, bh*h,
            linewidth=2, edgecolor=COLORS[cls % len(COLORS)], facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1-3, CLASS_NAMES[cls], color=COLORS[cls % len(COLORS)],
                fontsize=7, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.5))
    ax.set_title(title, fontsize=9)
    ax.axis("off")

def yolo_to_boxes(result):
    boxes = []
    if result.boxes is not None:
        for box in result.boxes:
            cls  = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy()
            cx   = ((xyxy[0]+xyxy[2])/2) / 640
            cy   = ((xyxy[1]+xyxy[3])/2) / 640
            bw   = (xyxy[2]-xyxy[0]) / 640
            bh   = (xyxy[3]-xyxy[1]) / 640
            boxes.append((cls, cx, cy, bw, bh))
    return boxes

def plot(baseline_path, gerber_path, data_root, fused_dir,
         output="paper/figures/fig3_qualitative.pdf"):
    model_base   = YOLO(baseline_path)
    model_gerber = YOLO(gerber_path)
    test_img_dir = Path(data_root) / "test" / "images"
    fused_img_dir = Path(fused_dir) / "test" / "images"
    test_lbl_dir = Path(data_root) / "test" / "labels"

    label_lookup = {}
    for lbl in test_lbl_dir.glob("*.txt"):
        label_lookup[get_base_stem(lbl.name)] = lbl

    def read_gt(img_path):
        base = get_base_stem(img_path.name)
        lbl  = label_lookup.get(base)
        boxes = []
        if lbl and lbl.exists():
            with open(lbl) as f:
                for line in f:
                    p = line.strip().split()
                    if p:
                        boxes.append((int(p[0]),
                                      float(p[1]),float(p[2]),
                                      float(p[3]),float(p[4])))
        return boxes

    fig, axes = plt.subplots(len(UNSEEN), 3, figsize=(15, 5*len(UNSEEN)))
    for row, fam in enumerate(UNSEEN):
        imgs = sorted(test_img_dir.glob(f"{fam}*.jpg"))
        if not imgs:
            continue
        img_path   = imgs[0]
        fused_path = fused_img_dir / img_path.name
        img_orig   = cv2.cvtColor(cv2.imread(str(img_path)),   cv2.COLOR_BGR2RGB)
        img_fused  = cv2.cvtColor(cv2.imread(str(fused_path)), cv2.COLOR_BGR2RGB)
        img_orig   = cv2.resize(img_orig,  (640,640))
        img_fused  = cv2.resize(img_fused, (640,640))
        gt_boxes   = read_gt(img_path)
        res_base   = model_base.predict(  str(img_path),   conf=0.3, verbose=False)[0]
        res_gerber = model_gerber.predict(str(fused_path), conf=0.3, verbose=False)[0]
        draw_boxes(axes[row,0], img_orig,  gt_boxes,              f"{fam} - Ground Truth")
        draw_boxes(axes[row,1], img_orig,  yolo_to_boxes(res_base),   f"Baseline ({len(res_base.boxes or [])} det)")
        draw_boxes(axes[row,2], img_fused, yolo_to_boxes(res_gerber), f"Gerber (ours) ({len(res_gerber.boxes or [])} det)")
    plt.suptitle("Qualitative Results on Unseen Board Families",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.savefig(output.replace(".pdf",".png"), dpi=300, bbox_inches="tight")
    print(f"Saved {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",  required=True)
    parser.add_argument("--gerber",    required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--fused_dir", required=True)
    parser.add_argument("--output", default="paper/figures/fig3_qualitative.pdf")
    args = parser.parse_args()
    plot(args.baseline, args.gerber, args.data_root, args.fused_dir, args.output)
