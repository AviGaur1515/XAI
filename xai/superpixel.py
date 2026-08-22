"""
=============================================================================
Brain Tumor MRI Classification & Explainable AI
Module: xai.superpixel
=============================================================================
Quantitative superpixel weight ranking and diverging heatmap generation
directly preserving the validated Kaggle notebook implementation.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_superpixel_bar_chart(explanation, pred_index=0, top_n=12):
    """
    Generates a high-resolution Matplotlib figure comparing top positive (supporting)
    and negative (contradicting) superpixel feature weights.
    Matches the Kaggle notebook superpixel weight bar chart.
    
    Args:
        explanation: LIME explanation object containing .local_exp and .segments.
        pred_index (int): Target predicted class index.
        top_n (int): Number of top superpixels to display.
        
    Returns:
        matplotlib.figure.Figure: Formatted 2-panel figure (Bar Chart & Weight Map).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    
    if explanation is None or not hasattr(explanation, "local_exp"):
        axes[0].text(0.5, 0.5, "No superpixel data available", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No weight map available", ha="center", va="center")
        plt.tight_layout()
        return fig

    local_weights = explanation.local_exp.get(pred_index, [])
    if not local_weights:
        axes[0].text(0.5, 0.5, f"No weights for class index {pred_index}", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No weight map", ha="center", va="center")
        plt.tight_layout()
        return fig

    # Sort superpixel weights by absolute magnitude
    weights = sorted(local_weights, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    seg_ids = [f"S{w[0]}" for w in weights]
    values = [w[1] for w in weights]
    colours = ["#2ca02c" if v > 0 else "#d62728" for v in values]

    # Panel 1: Horizontal Bar Chart
    axes[0].barh(seg_ids[::-1], values[::-1], color=colours[::-1], edgecolor="none", height=0.65)
    axes[0].axvline(0, color="#333333", linewidth=1.0, linestyle="--")
    axes[0].set_xlabel("Superpixel Contribution Weight (w)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Superpixel ID", fontsize=10, fontweight="bold")
    axes[0].set_title(f"Top {len(weights)} Influential Superpixels\n(Green = Supports Diagnosis, Red = Opposes)", fontsize=11, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.3, linestyle=":")
    axes[0].tick_params(labelsize=9)

    # Panel 2: Superpixel Diverging Weight Map
    lime_segs = explanation.segments
    w_dict = dict(local_weights)
    weight_map = np.zeros_like(lime_segs, dtype=np.float64)
    for seg_id, weight in w_dict.items():
        weight_map[lime_segs == seg_id] = weight

    vmax = float(np.max(np.abs(weight_map)))
    vmax = vmax if vmax > 1e-8 else 1.0

    im = axes[1].imshow(weight_map, cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Superpixel Weight Spatial Map", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Relative Weight", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    return fig
