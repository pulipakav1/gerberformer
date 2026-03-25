"""
Generate Figure 5: ablation study bar chart.
Usage: python plot_ablation.py --output paper/figures/fig5_ablation.pdf
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

ALPHAS      = [0.00, 0.10, 0.25, 0.50]
SEEN        = [0.9924, 0.9902, 0.9916, 0.9817]
UNSEEN      = [0.0588, 0.9890, 0.9928, 0.9821]
LABELS      = ["a=0\n(baseline)", "a=0.10", "a=0.25\n(ours)", "a=0.50"]

def plot(output="paper/figures/fig5_ablation.pdf"):
    x = np.arange(len(ALPHAS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x-w/2, SEEN,   w, label="Seen families",   color="#3498DB", alpha=0.85)
    ax.bar(x+w/2, UNSEEN, w, label="Unseen families", color="#E74C3C", alpha=0.85)
    ax.set_xlabel("Gerber Fusion Strength (alpha)", fontsize=12)
    ax.set_ylabel("mAP@50", fontsize=12)
    ax.set_title("Ablation: Effect of Gerber Prior Fusion Strength", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    for i, (s, u) in enumerate(zip(SEEN, UNSEEN)):
        ax.text(i-w/2, s+0.01, f"{s:.3f}", ha="center", fontsize=8)
        ax.text(i+w/2, u+0.01, f"{u:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.savefig(output.replace(".pdf",".png"), dpi=300, bbox_inches="tight")
    print(f"Saved {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="paper/figures/fig5_ablation.pdf")
    args = parser.parse_args()
    plot(args.output)
