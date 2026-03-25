"""
Train YOLOv8 baseline (image-only) on PCB defect dataset.
Usage: python train_baseline.py
"""
from ultralytics import YOLO
import yaml, os

DATA_ROOT = 'pcb-defect-dataset'
SAVE_DIR  = 'runs/baseline'

# Write dataset yaml
dataset_yaml = {
    'path':  DATA_ROOT,
    'train': 'train/images',
    'val':   'val/images',
    'test':  'test/images',
    'nc':    6,
    'names': ['missing_hole', 'mouse_bite', 'open_circuit',
              'short', 'spurious_copper', 'spur']
}
with open('pcb.yaml', 'w') as f:
    yaml.dump(dataset_yaml, f)

model = YOLO('yolov8s.pt')
model.train(
    data     = 'pcb.yaml',
    epochs   = 100,
    imgsz    = 640,
    batch    = 32,
    device   = 0,
    project  = SAVE_DIR,
    name     = 'yolov8_baseline',
    save     = True,
    patience = 20,
)
