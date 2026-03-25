"""
Per-family evaluation script.
Computes mAP@50 for each board family separately.
Usage: python evaluate.py --model path/to/best.pt --data pcb_fused.yaml --fused
"""
import argparse
from ultralytics import YOLO
from pathlib import Path
import yaml, os, shutil, json
import torch

def get_base_stem(filename):
    return Path(filename).stem.rsplit('_', 1)[0]

def eval_per_family(model_path, test_img_dir, test_lbl_dir, device='cpu'):
    model = YOLO(model_path)

    # Build label lookup (handles _600 vs _256 mismatch)
    label_lookup = {}
    for lbl in Path(test_lbl_dir).glob('*.txt'):
        label_lookup[get_base_stem(lbl.name)] = lbl

    # Build family list
    families = {}
    for img in Path(test_img_dir).glob('*.jpg'):
        parts = img.stem.split('_')
        fam   = f'{parts[0]}_{parts[1]}'
        if fam not in families:
            families[fam] = []
        families[fam].append(img)

    print(f'Found {len(families)} families')
    family_map = {}

    for family, imgs in sorted(families.items()):
        if len(imgs) < 5:
            print(f'  {family:<20} skipped (only {len(imgs)} images)')
            continue

        tmp = '/tmp/eval_tmp'
        os.makedirs(f'{tmp}/images', exist_ok=True)
        os.makedirs(f'{tmp}/labels', exist_ok=True)

        for img in imgs:
            shutil.copy(img, f'{tmp}/images/')
            base = get_base_stem(img.name)
            lbl  = label_lookup.get(base)
            if lbl:
                shutil.copy(lbl, Path(f'{tmp}/labels') / (img.stem + '.txt'))

        fam_yaml = {
            'path': tmp, 'train': 'images',
            'val': 'images', 'test': 'images',
            'nc': 6,
            'names': ['missing_hole','mouse_bite','open_circuit',
                      'short','spurious_copper','spur']
        }
        with open('/tmp/fam.yaml', 'w') as f:
            yaml.dump(fam_yaml, f)

        try:
            res   = model.val(data='/tmp/fam.yaml', split='test',
                              device=device, verbose=False)
            map50 = float(res.box.map50)
            family_map[family] = map50
            print(f'  {family:<20} mAP@50: {map50:.4f}  ({len(imgs)} images)')
        except Exception as e:
            print(f'  {family:<20} error: {e}')

        shutil.rmtree(tmp)

    vals        = list(family_map.values())
    seen        = ['l_light', 'rotation_270', 'rotation_90']
    unseen      = [f for f in family_map if f not in seen]
    seen_mean   = sum(family_map[f] for f in seen   if f in family_map) / max(len([f for f in seen   if f in family_map]), 1)
    unseen_mean = sum(family_map[f] for f in unseen if f in family_map) / max(len(unseen), 1)

    print(f'\nSeen families mean:   {seen_mean:.4f}')
    print(f'Unseen families mean: {unseen_mean:.4f}')
    print(f'Overall mean:         {sum(vals)/len(vals):.4f}')

    return family_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   type=str, required=True, help='path to best.pt')
    parser.add_argument('--imgdir',  type=str, required=True, help='test images dir')
    parser.add_argument('--lbldir',  type=str, required=True, help='test labels dir')
    parser.add_argument('--device',  type=str, default='0')
    parser.add_argument('--output',  type=str, default='family_results.json')
    args = parser.parse_args()

    results = eval_per_family(args.model, args.imgdir, args.lbldir, args.device)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {args.output}')
