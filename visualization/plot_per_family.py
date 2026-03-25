"""
Generate Figure 4: per-board-family comparison bar chart.
Usage: python plot_per_family.py --output paper/figures/fig4_per_family.pdf
"""
import argparse, json
import numpy as np
import matplotlib.pyplot as plt

BASELINE = {
    "l_light":0.9938,"light_01":0.1042,"light_04":0.0000,
    "light_05":0.0000,"light_06":0.1835,"light_07":0.0000,
    "light_08":0.0000,"light_09":0.1894,"light_10":0.1111,
    "light_11":0.0000,"light_12":0.0000,
    "rotation_270":0.9917,"rotation_90":0.9916,
}
GERBER = {
    "l_light":0.9939,"light_01":0.9950,"light_04":0.9950,
    "light_05":0.9950,"light_06":0.9829,"light_07":0.9950,
    "light_08":0.9950,"light_09":0.9950,"light_10":0.9950,
    "light_11":0.9855,"light_12":0.9950,
    "rotation_270":0.9909,"rotation_90":0.9899,
}
SEEN = {"l_light","rotation_270","rotation_90"}

def plot(output="paper/figures/fig4_per_family.pdf"):
    families = sorted(BASELINE.keys())
    x = np.arange(len(families))
    w = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x-w/2, [BASELINE[f] for f in families], w,
           label="YOLOv8 (image-only)", color="#FF6B6B", alpha=0.85)
    ax.bar(x+w/2, [GERBER[f]   for f in families], w,
           label="YOLOv8 + Gerber prior (ours)", color="#4ECDC4", alpha=0.85)
    for i, fam in enumerate(families):
        if fam not in SEEN:
            ax.axvspan(i-0.5, i+0.5, alpha=0.08, color="gray")
    ax.set_xlabel("Board Family", fontsize=12)
    ax.set_ylabel("mAP@50", fontsize=12)
    ax.set_title("Per-Board-Family Detection Performance\n"
                 "(shaded = unseen families during training)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.annotate("Unseen boards:\n0.06 -> 0.99",
                xy=(5, 0.5), fontsize=11, color="#2C3E50", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.9))
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.savefig(output.replace(".pdf",".png"), dpi=300, bbox_inches="tight")
    print(f"Saved {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="paper/figures/fig4_per_family.pdf")
    args = parser.parse_args()
    plot(args.output)
