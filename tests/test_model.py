import sys
sys.path.insert(0, "..")
import torch
from models.gerberformer_v1 import GerberFormer
from models.gerberformer_v2 import GerberFormerV2

def test_gerberformer_v1_forward():
    model = GerberFormer(pretrained=False)
    x     = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        out = model(x)
    assert len(out) == 3, "Expected 3 FPN scales"
    for o in out:
        assert "cls" in o and "reg" in o and "obj" in o and "unc" in o
    print("test_gerberformer_v1_forward PASSED")

def test_gerberformer_v2_forward():
    model  = GerberFormerV2(pretrained=False)
    images = torch.randn(2, 3, 640, 640)
    gerber = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        out = model(images, gerber)
    assert len(out) == 3
    print("test_gerberformer_v2_forward PASSED")

def test_gerberformer_v2_without_gerber():
    model  = GerberFormerV2(pretrained=False)
    images = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        out = model(images, gerber=None)
    assert len(out) == 3
    print("test_gerberformer_v2_without_gerber PASSED")

if __name__ == "__main__":
    test_gerberformer_v1_forward()
    test_gerberformer_v2_forward()
    test_gerberformer_v2_without_gerber()
    print("All model tests passed.")
