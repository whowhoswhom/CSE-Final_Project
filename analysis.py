"""
analysis.py - Plotting, training curves, misclassification mining, bbox overlays.

Provides:
  - plot_4panel(): the standard 4-subplot figure (train/val acc, train/val F1)
  - plot_training_curves(): loss/acc/F1 per epoch for a single run
  - find_misclassified(): mine misclassified samples from validation set
  - overlay_bbox_on_frame(): draw bbox + labels on original full frame
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path
from collections import defaultdict

from data_manifest import CLASSES

FIGURES_DIR = Path(__file__).parent / "figures"
ID_TO_CLASS = {v: k for k, v in CLASSES.items()}


# ═══════════════════════════════════════════════════════════════════════════
# 4-Panel Plot (required for every experiment)
# ═══════════════════════════════════════════════════════════════════════════

def plot_4panel(results, x_values, x_label, title="", x_log=False,
                save_name=None):
    """
    Generate the standard 4-subplot figure.

    Args:
        results: list of dicts, each with keys:
            'label', 'best_train_acc', 'best_val_acc', 'best_train_f1', 'best_val_f1'
        x_values: list of x-axis values (one per result)
        x_label: x-axis label string
        title: overall figure title
        x_log: use log scale on x-axis
        save_name: filename to save in figures/
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
              "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    metrics = [
        ("best_train_acc", "Training Accuracy"),
        ("best_val_acc", "Validation Accuracy"),
        ("best_train_f1", "Training F1 (weighted)"),
        ("best_val_f1", "Validation F1 (weighted)"),
    ]

    for ax_idx, (key, ylabel) in enumerate(metrics):
        ax = axes[ax_idx // 2][ax_idx % 2]
        values = [r[key] for r in results]
        labels = [r.get("label", str(x)) for r, x in zip(results, x_values)]

        for i, (x, y, lbl) in enumerate(zip(x_values, values, labels)):
            color = colors[i % len(colors)]
            ax.plot(x, y, "o-", color=color, markersize=8, label=lbl)

        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if x_log:
            ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_name:
        FIGURES_DIR.mkdir(exist_ok=True)
        fig.savefig(FIGURES_DIR / save_name, dpi=150, bbox_inches="tight")
        print(f"  Saved: {FIGURES_DIR / save_name}")

    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Training Curves (single run)
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(history, title="Training Curves", save_name=None):
    """Plot loss, accuracy, and F1 curves from a training history dict."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    if history.get("best_epoch"):
        axes[0].axvline(history["best_epoch"], color="red", linestyle="--",
                        alpha=0.5, label=f"Best (ep {history['best_epoch']})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    if history.get("best_epoch"):
        axes[1].axvline(history["best_epoch"], color="red", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1
    axes[2].plot(epochs, history["train_f1"], label="Train")
    axes[2].plot(epochs, history["val_f1"], label="Val")
    if history.get("best_epoch"):
        axes[2].axvline(history["best_epoch"], color="red", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Weighted F1")
    axes[2].set_title("Weighted F1")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_name:
        FIGURES_DIR.mkdir(exist_ok=True)
        fig.savefig(FIGURES_DIR / save_name, dpi=150, bbox_inches="tight")

    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Results Summary Table
# ═══════════════════════════════════════════════════════════════════════════

def print_results_table(results):
    """Print a compact results table."""
    header = f"{'Label':<20} {'Params':>10} {'Best Ep':>8} {'Train Acc':>10} {'Val Acc':>10} {'Train F1':>10} {'Val F1':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        best_epoch = str(r.get("best_epoch", 0))
        print(f"{r.get('label','?'):<20} "
              f"{r.get('param_count', 0):>10,} "
              f"{best_epoch:>8} "
              f"{r.get('best_train_acc', 0):>10.4f} "
              f"{r.get('best_val_acc', 0):>10.4f} "
              f"{r.get('best_train_f1', 0):>10.4f} "
              f"{r.get('best_val_f1', 0):>10.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Misclassification Mining
# ═══════════════════════════════════════════════════════════════════════════

def find_misclassified(preds, labels, manifest_df, n_per_class=2):
    """
    Find misclassified samples from the validation set.

    Args:
        preds: numpy array of predicted labels
        labels: numpy array of true labels
        manifest_df: the validation DataFrame with patch_path, class_name, etc.
        n_per_class: number of misclassified samples to find per class

    Returns:
        dict: {class_name: [(idx, true_class, pred_class, row), ...]}
    """
    misclassified = defaultdict(list)
    df = manifest_df.reset_index(drop=True)

    for i in range(len(preds)):
        if preds[i] != labels[i]:
            true_class = ID_TO_CLASS[labels[i]]
            pred_class = ID_TO_CLASS[preds[i]]
            if len(misclassified[true_class]) < n_per_class:
                misclassified[true_class].append({
                    "idx": i,
                    "true_class": true_class,
                    "pred_class": pred_class,
                    "patch_path": df.iloc[i]["patch_path"],
                    "source_frame_path": df.iloc[i].get("source_frame_path", ""),
                    "bbox_xywh": df.iloc[i].get("bbox_xywh", ""),
                    "video_id": df.iloc[i].get("video_id", ""),
                    "frame_index": df.iloc[i].get("frame_index", ""),
                })

    return dict(misclassified)


def plot_misclassified(misclassified_dict, title="Misclassified Samples",
                       save_name=None):
    """Display misclassified patch images in a grid."""
    all_items = []
    for cls, items in sorted(misclassified_dict.items()):
        all_items.extend(items)

    if not all_items:
        print("No misclassified samples found.")
        return

    n = len(all_items)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i, item in enumerate(all_items):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        try:
            img = Image.open(item["patch_path"]).convert("RGB")
            ax.imshow(img)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "Image\nnot found", ha="center", va="center")
        ax.set_title(f"True: {item['true_class']}\nPred: {item['pred_class']}",
                     fontsize=10, color="red")
        ax.axis("off")

    # Hide unused axes
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_name:
        FIGURES_DIR.mkdir(exist_ok=True)
        fig.savefig(FIGURES_DIR / save_name, dpi=150, bbox_inches="tight")

    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Full-Frame BBox Overlay (Part 5)
# ═══════════════════════════════════════════════════════════════════════════

def overlay_bbox_on_frame(frame_path, bbox_xywh, true_label, pred_label,
                          ax=None):
    """
    Draw a bounding box on the original full frame.

    Args:
        frame_path: path to the full frame image
        bbox_xywh: [x, y, w, h] bounding box
        true_label: ground truth class name
        pred_label: predicted class name
        ax: matplotlib axis (creates one if None)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    try:
        img = Image.open(frame_path).convert("RGB")
        ax.imshow(img)
    except FileNotFoundError:
        ax.text(0.5, 0.5, f"Frame not found:\n{frame_path}",
                ha="center", va="center", fontsize=10)
        return ax

    # Parse bbox if it's a string
    if isinstance(bbox_xywh, str):
        bbox_xywh = eval(bbox_xywh)

    x, y, w, h = bbox_xywh
    color = "green" if true_label == pred_label else "red"
    rect = mpatches.Rectangle((x, y), w, h, linewidth=2,
                               edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    ax.text(x, y - 5, f"True: {true_label} | Pred: {pred_label}",
            fontsize=9, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    ax.axis("off")
    return ax
