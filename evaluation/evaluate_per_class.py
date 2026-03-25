"""
Per-class AP evaluation comparing baseline vs Gerber model.
Usage: python evaluate_per_class.py
       --baseline path/to/baseline/best.pt
       --gerber   path/to/gerber/best.pt
       --data     data/yamls/pcb_test.yaml
"""
import argparse, json
from ultralytics import YOLO

CLASS_NAMES = ["missing_hole","mouse_bite","open_circuit",
               "short","spurious_copper","spur"]

def eval_per_class(model_path, data_yaml, device=0):
    model   = YOLO(model_path)
    results = model.val(data=data_yaml, split="test",
                        device=device, verbose=False)
    return {name: float(ap)
            for name, ap in zip(CLASS_NAMES, results.box.ap50)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--gerber",   required=True)
    parser.add_argument("--data",     required=True)
    parser.add_argument("--device",   default="0")
    parser.add_argument("--output",   default="per_class_results.json")
    args = parser.parse_args()

    print("Evaluating baseline...")
    base_ap = eval_per_class(args.baseline, args.data, args.device)
    print("Evaluating Gerber model...")
    gerb_ap = eval_per_class(args.gerber, args.data, args.device)

    print(f"{'Class':<20} {'Baseline':>10} {'Gerber':>10} {'Delta':>10}")
    print("-" * 52)
    for cls in CLASS_NAMES:
        delta = gerb_ap[cls] - base_ap[cls]
        print(f"  {cls:<20} {base_ap[cls]:>10.4f} {gerb_ap[cls]:>10.4f} {delta:>+10.4f}")

    out = {"baseline": base_ap, "gerber": gerb_ap}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {args.output}")
