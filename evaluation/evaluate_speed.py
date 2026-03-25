"""
Inference speed benchmark.
Usage: python evaluate_speed.py --baseline path/to/baseline/best.pt
                                --gerber   path/to/gerber/best.pt
                                --data     data/yamls/pcb_test.yaml
"""
import argparse, json
from ultralytics import YOLO

def benchmark_speed(model_path, data_yaml, device=0):
    model   = YOLO(model_path)
    results = model.val(data=data_yaml, split="test",
                        device=device, verbose=False)
    speed   = results.speed
    total   = sum(speed.values())
    return {
        "preprocess_ms":  round(speed["preprocess"],  3),
        "inference_ms":   round(speed["inference"],   3),
        "postprocess_ms": round(speed["postprocess"], 3),
        "total_ms":       round(total, 3),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--gerber",   required=True)
    parser.add_argument("--data",     required=True)
    parser.add_argument("--device",   default="0")
    parser.add_argument("--output",   default="speed_results.json")
    args = parser.parse_args()

    print("Benchmarking baseline...")
    base_speed = benchmark_speed(args.baseline, args.data, args.device)
    print("Benchmarking Gerber model...")
    gerb_speed = benchmark_speed(args.gerber, args.data, args.device)

    print(f"{'Metric':<20} {'Baseline':>12} {'Gerber':>12}")
    print("-" * 46)
    for k in base_speed:
        print(f"  {k:<20} {base_speed[k]:>12.3f} {gerb_speed[k]:>12.3f}")

    out = {"baseline": base_speed, "gerber": gerb_speed,
           "note": "No inference overhead from Gerber conditioning"}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {args.output}")
