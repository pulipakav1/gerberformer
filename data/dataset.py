import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEFECT_CLASSES = [
    "missing_hole", "mouse_bite", "open_circuit",
    "short", "spurious_copper", "spur",
]
CLASS2IDX  = {c: i for i, c in enumerate(DEFECT_CLASSES)}
IDX2CLASS  = {i: c for c, i in CLASS2IDX.items()}
NUM_CLASSES = len(DEFECT_CLASSES)

def parse_board_family(filename):
    stem  = Path(filename).stem
    parts = stem.split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else parts[0]

def read_yolo_labels(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append({"class_id": class_id, "cx": cx, "cy": cy, "bw": bw, "bh": bh})
    return boxes

def get_train_transforms(img_size=640):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3))

def get_val_transforms(img_size=640):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3))

class PCBDefectDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=640, transforms=None):
        assert split in ("train", "val", "test")
        self.root_dir  = Path(root_dir)
        self.split     = split
        self.img_size  = img_size
        self.img_dir   = self.root_dir / split / "images"
        self.lbl_dir   = self.root_dir / split / "labels"
        self.img_paths = sorted([
            p for p in self.img_dir.glob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        self.transforms = transforms or (
            get_train_transforms(img_size) if split == "train"
            else get_val_transforms(img_size)
        )
        print(f"[Dataset] {split}: {len(self.img_paths)} images")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img      = cv2.imread(str(img_path))
        img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]
        lbl_path     = self.lbl_dir / (img_path.stem + ".txt")
        raw_boxes    = read_yolo_labels(str(lbl_path), img_w, img_h)
        bboxes_yolo  = [[b["cx"], b["cy"], b["bw"], b["bh"]] for b in raw_boxes]
        class_labels = [b["class_id"] for b in raw_boxes]
        try:
            aug = self.transforms(image=img, bboxes=bboxes_yolo, class_labels=class_labels)
        except Exception:
            aug = get_val_transforms(self.img_size)(image=img, bboxes=bboxes_yolo, class_labels=class_labels)
        img          = aug["image"]
        bboxes_yolo  = list(aug["bboxes"])
        class_labels = list(aug["class_labels"])
        num_boxes    = len(bboxes_yolo)
        targets = torch.zeros((num_boxes, 5), dtype=torch.float32)
        for i, (box, cls) in enumerate(zip(bboxes_yolo, class_labels)):
            targets[i] = torch.tensor([cls, box[0], box[1], box[2], box[3]])
        return {
            "image":        img,
            "targets":      targets,
            "image_path":   str(img_path),
            "board_family": parse_board_family(img_path.name),
            "num_boxes":    num_boxes,
        }

def collate_fn(batch):
    images    = torch.stack([b["image"] for b in batch])
    max_boxes = max(max(b["num_boxes"] for b in batch), 1)
    targets_padded = torch.zeros((len(batch), max_boxes, 5))
    targets_padded[:, :, 0] = -1
    for i, b in enumerate(batch):
        n = b["num_boxes"]
        if n > 0:
            targets_padded[i, :n] = b["targets"]
    return {
        "image":          images,
        "targets":        targets_padded,
        "image_paths":    [b["image_path"]   for b in batch],
        "board_families": [b["board_family"] for b in batch],
        "num_boxes":      [b["num_boxes"]    for b in batch],
    }
