import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEFECT_CLASSES = [
    'missing_hole', 'mouse_bite', 'open_circuit',
    'short', 'spurious_copper', 'spur',
]
CLASS2IDX  = {c: i for i, c in enumerate(DEFECT_CLASSES)}
IDX2CLASS  = {i: c for c, i in CLASS2IDX.items()}
NUM_CLASSES = len(DEFECT_CLASSES)


def parse_board_family(filename):
    stem  = Path(filename).stem
    parts = stem.split('_')
    return f'{parts[0]}_{parts[1]}' if len(parts) >= 2 else parts[0]


def read_yolo_labels(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts    = line.split()
            class_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), \
                              float(parts[3]), float(parts[4])
            boxes.append({'class_id': class_id,
                          'cx': cx, 'cy': cy, 'bw': bw, 'bh': bh})
    return boxes


def get_transforms(split='train', img_size=640):
    if split == 'train':
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size,
                          border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.2, hue=0.1, p=0.5),
            A.GaussNoise(p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='yolo', label_fields=['class_labels'],
            min_visibility=0.3))
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size,
                          border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='yolo', label_fields=['class_labels'],
            min_visibility=0.3))


class PCBDefectDatasetV2(Dataset):
    """
    Phase 2 dataset — loads AOI image + synthetic Gerber prior.
    The Gerber prior is the median copper mask for the board family.
    """

    def __init__(self, root_dir, prior_dir, split='train',
                 img_size=640, transforms=None):
        assert split in ('train', 'val', 'test')
        self.root_dir  = Path(root_dir)
        self.prior_dir = Path(prior_dir)
        self.split     = split
        self.img_size  = img_size
        self.img_dir   = self.root_dir / split / 'images'
        self.lbl_dir   = self.root_dir / split / 'labels'

        self.img_paths = sorted([
            p for p in self.img_dir.glob('*')
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ])
        self.transforms = transforms or get_transforms(split, img_size)

        # Load all priors into memory
        self.priors = {}
        for prior_file in self.prior_dir.glob('*_prior.png'):
            fam = prior_file.stem.replace('_prior', '')
            img = cv2.imread(str(prior_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.priors[fam] = img

        print(f'[DatasetV2] {split}: {len(self.img_paths)} images | '
              f'{len(self.priors)} priors loaded')

    def __len__(self):
        return len(self.img_paths)

    def _normalize_prior(self, prior_img):
        """Normalize prior image to tensor."""
        prior = prior_img.astype(np.float32) / 255.0
        mean  = np.array([0.485, 0.456, 0.406])
        std   = np.array([0.229, 0.224, 0.225])
        prior = (prior - mean) / std
        return torch.from_numpy(prior).permute(2, 0, 1).float()

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img      = cv2.imread(str(img_path))
        img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        lbl_path     = self.lbl_dir / (img_path.stem + '.txt')
        raw_boxes    = read_yolo_labels(str(lbl_path), img_w, img_h)
        bboxes_yolo  = [[b['cx'], b['cy'], b['bw'], b['bh']]
                        for b in raw_boxes]
        class_labels = [b['class_id'] for b in raw_boxes]

        try:
            aug = self.transforms(image=img, bboxes=bboxes_yolo,
                                  class_labels=class_labels)
        except Exception:
            aug = get_transforms('val', self.img_size)(
                image=img, bboxes=bboxes_yolo,
                class_labels=class_labels)

        img          = aug['image']
        bboxes_yolo  = list(aug['bboxes'])
        class_labels = list(aug['class_labels'])
        num_boxes    = len(bboxes_yolo)

        targets = torch.zeros((num_boxes, 5), dtype=torch.float32)
        for i, (box, cls) in enumerate(zip(bboxes_yolo, class_labels)):
            targets[i] = torch.tensor([cls, box[0], box[1],
                                        box[2], box[3]])

        # Get Gerber prior for this board family
        family = parse_board_family(img_path.name)
        prior  = self.priors.get(family)

        if prior is None:
            # Fallback: use first available prior
            prior = next(iter(self.priors.values()))

        prior_tensor = self._normalize_prior(prior)

        return {
            'image':        img,
            'gerber':       prior_tensor,
            'targets':      targets,
            'image_path':   str(img_path),
            'board_family': family,
            'num_boxes':    num_boxes,
        }


def collate_fn_v2(batch):
    images  = torch.stack([b['image']  for b in batch])
    gerbers = torch.stack([b['gerber'] for b in batch])

    max_boxes = max(max(b['num_boxes'] for b in batch), 1)
    targets_padded = torch.zeros((len(batch), max_boxes, 5))
    targets_padded[:, :, 0] = -1
    for i, b in enumerate(batch):
        n = b['num_boxes']
        if n > 0:
            targets_padded[i, :n] = b['targets']

    return {
        'image':          images,
        'gerber':         gerbers,
        'targets':        targets_padded,
        'image_paths':    [b['image_path']   for b in batch],
        'board_families': [b['board_family'] for b in batch],
        'num_boxes':      [b['num_boxes']    for b in batch],
    }
