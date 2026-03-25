# GerberFormer: Design-Conditioned PCB Defect Detection

## Overview
This repository implements design-conditioned PCB defect detection
using synthetic Gerber priors. The key finding is that image-only
detectors fail catastrophically on unseen board families (mAP@50: 0.06)
while Gerber-conditioned models maintain near-perfect performance (mAP@50: 0.99).

## Key Results

| Model                  | Seen mAP@50 | Unseen mAP@50 |
|------------------------|-------------|---------------|
| YOLOv8 (image-only)    | 0.9924      | 0.0588        |
| YOLOv8 + Gerber (ours) | 0.9916      | 0.9928        |

## Project Structure

    gerberformer/
    configs/          Hyperparameter configs for all experiments
    data/             Dataset loaders and YAML configs
    models/           Model architectures (GerberFormerV1, V2)
    priors/           Synthetic Gerber prior construction
    training/         Training scripts
    evaluation/       Evaluation and metrics scripts
    visualization/    Figure generation
    experiments/      Results per experiment
    notebooks/        Exploratory notebooks
    tests/            Unit tests
    paper/            Figures, tables, results

## Installation

    pip install -r requirements.txt
    pip install -e .

## Reproduce Results

Step 1: Download dataset
    kaggle datasets download -d norbertelter/pcb-defect-dataset
    unzip pcb-defect-dataset.zip -d pcb-defect-dataset

Step 2: Build Gerber priors
    python priors/build_priors.py

Step 3: Build fused dataset
    python priors/build_fused_dataset.py

Step 4: Train baseline
    python training/train_baseline.py

Step 5: Train Gerber model
    python training/train_gerber.py

Step 6: Evaluate per family
    python evaluation/evaluate_per_family.py
        --model experiments/gerber_alpha025/weights/best.pt
        --imgdir pcb-fused-dataset/test/images
        --lbldir pcb-defect-dataset/test/labels

Step 7: Run ablation
    python training/train_ablation.py

## Run Tests

    python tests/test_dataset.py
    python tests/test_model.py
    python tests/test_priors.py

## Dataset
PKU PCB Defect Dataset
https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset
10,664 images | 6 defect classes | 13 board families

## Citation
Coming soon.

## License
MIT
