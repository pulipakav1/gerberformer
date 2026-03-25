
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import os

def build_gerber_priors(train_img_dir, output_dir, img_size=640, n_samples=100):
    os.makedirs(output_dir, exist_ok=True)
    families = defaultdict(list)
    for img in Path(train_img_dir).glob("*.jpg"):
        parts = img.stem.split("_")
        fam = f"{parts[0]}_{parts[1]}"
        families[fam].append(img)
    for fam, img_paths in sorted(families.items()):
        print(f"Building prior for {fam} ({len(img_paths)} images)...")
        sample = img_paths[:n_samples]
        stack  = []
        for p in sample:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            stack.append(img.astype(np.float32))
        if not stack:
            continue
        median   = np.median(np.stack(stack), axis=0).astype(np.uint8)
        gray     = cv2.cvtColor(median, cv2.COLOR_RGB2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 5)
        sobelx   = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobely   = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        gradient = cv2.normalize(gradient, None, 0, 255,
                                  cv2.NORM_MINMAX).astype(np.uint8)
        prior = np.stack([enhanced, adaptive, gradient], axis=-1)
        cv2.imwrite(os.path.join(output_dir, f"{fam}_prior.png"), prior)
        print(f"  Saved {fam}_prior.png")

if __name__ == "__main__":
    build_gerber_priors(
        train_img_dir="pcb-defect-dataset/train/images",
        output_dir="gerber_priors",
    )
