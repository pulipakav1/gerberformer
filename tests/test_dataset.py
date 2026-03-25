import sys
sys.path.insert(0, "..")
import torch
from data.dataset import PCBDefectDataset, collate_fn
from torch.utils.data import DataLoader

def test_dataset_loads():
    ds = PCBDefectDataset("pcb-defect-dataset", split="train", img_size=640)
    assert len(ds) > 0, "Dataset is empty"
    sample = ds[0]
    assert "image"   in sample
    assert "targets" in sample
    assert sample["image"].shape == torch.Size([3, 640, 640])
    print(f"test_dataset_loads PASSED ({len(ds)} images)")

def test_collate_fn():
    ds     = PCBDefectDataset("pcb-defect-dataset", split="train", img_size=640)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn, num_workers=0)
    batch  = next(iter(loader))
    assert batch["image"].shape[0] == 4
    assert batch["targets"].shape[0] == 4
    print("test_collate_fn PASSED")

def test_board_families():
    ds      = PCBDefectDataset("pcb-defect-dataset", split="train", img_size=640)
    sample  = ds[0]
    assert "board_family" in sample
    assert len(sample["board_family"]) > 0
    print(f"test_board_families PASSED (family: {sample['board_family']})")

if __name__ == "__main__":
    test_dataset_loads()
    test_collate_fn()
    test_board_families()
    print("All dataset tests passed.")
