
import cv2
import numpy as np
import shutil
from pathlib import Path
import os

def build_fused_dataset(data_root, prior_dir, output_dir, alpha=0.25):
    priors = {}
    for pf in Path(prior_dir).glob("*_prior.png"):
        fam = pf.stem.replace("_prior", "")
        img = cv2.imread(str(pf))
        priors[fam] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Loaded {len(priors)} priors")

    def get_family(filename):
        parts = Path(filename).stem.split("_")
        return f"{parts[0]}_{parts[1]}"

    def get_base_stem(filename):
        return Path(filename).stem.rsplit("_", 1)[0]

    for split in ["train", "val", "test"]:
        img_dir = Path(data_root) / split / "images"
        lbl_dir = Path(data_root) / split / "labels"
        out_img = Path(output_dir) / split / "images"
        out_lbl = Path(output_dir) / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        imgs = list(img_dir.glob("*.jpg"))
        print(f"Processing {split}: {len(imgs)} images...")
        for img_path in imgs:
            fam   = get_family(img_path.name)
            prior = priors.get(fam, next(iter(priors.values())))
            img   = cv2.imread(str(img_path))
            img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img   = cv2.resize(img, (640, 640))
            fused = cv2.addWeighted(
                img.astype(np.float32),   1 - alpha,
                prior.astype(np.float32), alpha, 0
            ).astype(np.uint8)
            cv2.imwrite(str(out_img / img_path.name),
                        cv2.cvtColor(fused, cv2.COLOR_RGB2BGR))
            base = get_base_stem(img_path.name)
            lbls = list(lbl_dir.glob(f"{base}_*.txt"))
            if lbls:
                shutil.copy(lbls[0], out_lbl / (img_path.stem + ".txt"))
        print(f"  {split} done.")

if __name__ == "__main__":
    build_fused_dataset(
        data_root  = "pcb-defect-dataset",
        prior_dir  = "gerber_priors",
        output_dir = "pcb-fused-dataset",
        alpha      = 0.25,
    )
