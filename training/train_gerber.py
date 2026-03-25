"""
Train YOLOv8 + Gerber fusion on PCB defect dataset.
Run build_gerber_priors.py and build_fused_dataset.py first.
Usage: python train_gerber.py
"""
from ultralytics import YOLO
import yaml

FUSED_DIR = 'pcb-fused-dataset'
SAVE_DIR  = 'runs/gerber'

dataset_yaml = {
    'path':  FUSED_DIR,
    'train': 'train/images',
    'val':   'val/images',
    'test':  'test/images',
    'nc':    6,
    'names': ['missing_hole', 'mouse_bite', 'open_circuit',
              'short', 'spurious_copper', 'spur']
}
with open('pcb_fused.yaml', 'w') as f:
    yaml.dump(dataset_yaml, f)

model = YOLO('yolov8s.pt')
model.train(
    data     = 'pcb_fused.yaml',
    epochs   = 100,
    imgsz    = 640,
    batch    = 32,
    device   = 0,
    project  = SAVE_DIR,
    name     = 'yolov8_gerber',
    save     = True,
    patience = 20,
)
