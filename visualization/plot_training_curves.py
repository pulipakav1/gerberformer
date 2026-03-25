"""
Plot training loss curves from history.json.
Usage: python plot_training_curves.py --history path/to/history.json
"""
import argparse, json
import matplotlib.pyplot as plt

def plot(history_path, output="paper/figures/training_curves.pdf"):
    with open(history_path) as f:
        history = json.load(f)
    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_loss,   label="Val Loss",   linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GerberFormer Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.savefig(output.replace(".pdf",".png"), dpi=300, bbox_inches="tight")
    print(f"Saved {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--output",  default="paper/figures/training_curves.pdf")
    args = parser.parse_args()
    plot(args.history, args.output)
