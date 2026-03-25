"""
Train ablation models for different alpha values.
Usage: python train_ablation.py
"""
from ultralytics import YOLO
from build_fused_dataset import build_fused_dataset
import yaml, os

ALPHAS    = [0.10, 0.25, 0.50]
DATA_ROOT = 'pcb-defect-dataset'
PRIOR_DIR = 'gerber_priors'

for alpha in ALPHAS:
    print(f'\n{"="*50}')
    print(f'Training alpha={alpha}')
    print(f'{"="*50}')

    out_dir = f'pcb-fused-alpha{int(alpha*100)}'
    if not os.path.exists(out_dir):
        build_fused_dataset(DATA_ROOT, PRIOR_DIR, out_dir, alpha=alpha)

    dataset_yaml = {
        'path':  out_dir,
        'train': 'train/images',
        'val':   'val/images',
        'test':  'test/images',
        'nc':    6,
        'names': ['missing_hole','mouse_bite','open_circuit',
                  'short','spurious_copper','spur']
    }
    yaml_path = f'pcb_alpha{int(alpha*100)}.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f)

    model = YOLO('yolov8s.pt')
    model.train(
        data     = yaml_path,
        epochs   = 50,
        imgsz    = 640,
        batch    = 32,
        device   = 0,
        project  = 'runs/ablation',
        name     = f'yolov8_alpha{int(alpha*100)}',
        save     = True,
        patience = 15,
        verbose  = False,
    )
    print(f'alpha={alpha} done.')
