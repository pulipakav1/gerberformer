"""
Overall test set evaluation.
Usage: python evaluate_overall.py --model path/to/best.pt --data data/yamls/pcb_test.yaml
"""
import argparse, json
from ultralytics import YOLO

def evaluate_overall(model_path, data_yaml, device=0, split="test"):
    model   = YOLO(model_path)
    results = model.val(data=data_yaml, split=split, device=device)
    out = {
        "mAP50":     float(results.box.map50),
        "mAP50_95":  float(results.box.map),
        "precision": float(results.box.mp),
        "recall":    float(results.box.mr),
        "speed":     results.speed,
        "per_class_ap50": {
            name: float(ap)
            for name, ap in zip(
                ["missing_hole","mouse_bite","open_circuit",
                 "short","spurious_copper","spur"],
                results.box.ap50
            )
        }
    }
    print(f"mAP@50:    {out['mAP50']:.4f}")
    print(f"mAP@50-95: {out['mAP50_95']:.4f}")
    print("Per-class AP@50:")
    for cls, ap in out["per_class_ap50"].items():
        print(f"  {cls:<20} {ap:.4f}")
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True)
    parser.add_argument("--data",   required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default="overall_results.json")
    args = parser.parse_args()
    results = evaluate_overall(args.model, args.data, args.device)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")
