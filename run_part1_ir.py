"""
Run Part 1 IR experiments: 4 CNN architectures on thermal data.
Saves results, checkpoints, and figures.
"""
import torch
import json
import time
import os
import atexit
from pathlib import Path
from torchinfo import summary

from models import build_cnn, verify_10x_rule
from training import get_dataloaders, train_model
from analysis import plot_4panel, plot_training_curves, print_results_table

MANIFEST = "manifests/ir_manifest.csv"
MODALITY = "IR"
BATCH_SIZE = 256
EPOCHS = 70
LR = 1e-3
MIN_EPOCHS = 50
PATIENCE = 10
LOCK_PATH = Path("checkpoints") / "run_part1_ir.lock"


def _pid_exists(pid: int) -> bool:
    """Return True if a process with this PID appears to be alive."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock() -> None:
    """
    Prevent duplicate Part 1 IR runs from competing for the same GPU/checkpoints.
    Removes stale lock files from interrupted runs.
    """
    LOCK_PATH.parent.mkdir(exist_ok=True)

    if LOCK_PATH.exists():
        try:
            existing_pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
        except Exception:
            existing_pid = -1

        if _pid_exists(existing_pid):
            raise RuntimeError(
                f"Another Part 1 IR run is already active (PID {existing_pid})."
            )
        LOCK_PATH.unlink(missing_ok=True)

    LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
    atexit.register(lambda: LOCK_PATH.unlink(missing_ok=True))

def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # 1. Verify 10x rule
    print("=" * 60)
    print("Verifying 10x parameter growth rule")
    print("=" * 60)
    param_counts = verify_10x_rule()
    print()

    # 2. Load data
    print("Loading IR data...")
    train_loader, val_loader, train_df, val_df = get_dataloaders(
        MANIFEST, batch_size=BATCH_SIZE, num_workers=4
    )
    print(f"Train: {len(train_loader.dataset):,} samples ({len(train_loader)} batches)")
    print(f"Val:   {len(val_loader.dataset):,} samples ({len(val_loader)} batches)")
    print()

    # 3. Train all architectures
    results = []
    histories = {}
    total_start = time.time()

    for arch in ["A", "B", "C", "D"]:
        print(f"\n{'#' * 60}")
        print(f"# Training Architecture {arch} ({param_counts[arch]:,} params)")
        print(f"{'#' * 60}")

        model, count = build_cnn(arch)

        # Print param count (full torchinfo summary available in notebook)
        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params: {total_p:,} | Trainable: {train_p:,}")

        arch_start = time.time()
        history = train_model(
            model, train_loader, val_loader,
            epochs=EPOCHS, lr=LR,
            patience=PATIENCE, min_epochs=MIN_EPOCHS,
            checkpoint_name=f"part1_ir_arch{arch}"
        )
        arch_time = time.time() - arch_start
        histories[arch] = history

        results.append({
            "label": f"Arch {arch}",
            "arch": arch,
            "param_count": count,
            "best_epoch": history["best_epoch"],
            "best_train_acc": history["best_train_acc"],
            "best_val_acc": history["best_val_acc"],
            "best_train_f1": history["best_train_f1"],
            "best_val_f1": history["best_val_f1"],
            "train_time_min": arch_time / 60,
        })

        print(f"\n  Time: {arch_time / 60:.1f} min")
        del model
        torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # 4. Results
    print(f"\n{'=' * 60}")
    print(f"PART 1 IR - RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print_results_table(results)

    # 5. Identify optimal and overfitting candidate
    best = max(results, key=lambda r: r["best_val_f1"])
    print(f"\nOptimal architecture: {best['label']}")
    print(f"  Val Acc: {best['best_val_acc']:.4f}, Val F1: {best['best_val_f1']:.4f}")

    print("\nOverfitting analysis (train - val gap):")
    for r in results:
        gap_acc = r["best_train_acc"] - r["best_val_acc"]
        gap_f1 = r["best_train_f1"] - r["best_val_f1"]
        print(f"  {r['label']}: acc gap = {gap_acc:.4f}, F1 gap = {gap_f1:.4f}")

    print(f"\nTotal training time: {total_time / 60:.1f} min")

    # 6. Save figures
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend

    x_values = [r["param_count"] for r in results]
    plot_4panel(
        results, x_values,
        x_label="Number of Parameters",
        title=f"Part 1 - {MODALITY}: CNN Architecture Comparison",
        x_log=True,
        save_name=f"part1_{MODALITY.lower()}_architectures.png"
    )

    for arch in ["A", "B", "C", "D"]:
        plot_training_curves(
            histories[arch],
            title=f"{MODALITY} Architecture {arch} ({param_counts[arch]:,} params)",
            save_name=f"part1_{MODALITY.lower()}_arch{arch}_curves.png"
        )

    # 7. Save raw results for later use
    results_path = Path("figures") / "part1_ir_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save histories for Part 2 reference
    histories_path = Path("figures") / "part1_ir_histories.json"
    hist_serializable = {}
    for arch, h in histories.items():
        hist_serializable[arch] = {
            k: (v if not isinstance(v, list) else [float(x) for x in v])
            for k, v in h.items()
        }
    with open(histories_path, "w") as f:
        json.dump(hist_serializable, f, indent=2)

    return results, histories


if __name__ == "__main__":
    acquire_lock()
    main()
