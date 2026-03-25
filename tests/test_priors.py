import sys
sys.path.insert(0, "..")
import os
import cv2
import numpy as np
from priors.build_priors import build_gerber_priors

def test_prior_shape():
    build_gerber_priors(
        train_img_dir="pcb-defect-dataset/train/images",
        output_dir="/tmp/test_priors",
        n_samples=5,
    )
    priors = list(os.listdir("/tmp/test_priors"))
    assert len(priors) > 0, "No priors built"
    prior = cv2.imread(f"/tmp/test_priors/{priors[0]}")
    assert prior.shape == (640, 640, 3), f"Wrong shape: {prior.shape}"
    print(f"test_prior_shape PASSED ({len(priors)} priors)")

def test_prior_not_blank():
    priors = list(os.listdir("/tmp/test_priors"))
    for fname in priors:
        img = cv2.imread(f"/tmp/test_priors/{fname}")
        assert img.std() > 5, f"Prior {fname} looks blank (std={img.std():.2f})"
    print("test_prior_not_blank PASSED")

if __name__ == "__main__":
    test_prior_shape()
    test_prior_not_blank()
    print("All prior tests passed.")
