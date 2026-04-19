"""
Resume Part 2 RGB experiments without retraining completed sections.

- Reconstructs 2A/2B metrics from existing checkpoints
- Trains or reloads 2C/2D as needed
- Saves figures and a JSON summary
"""
import atexit
import json
import os
import time
from pathlib import Path

import matplotlib
import torch

from analysis import plot_4panel, print_results_table
from models import build_cnn
from training import (
    evaluate_model,
    get_augmentation,
    get_dataloaders,
    train_model,
)

matplotlib.use("Agg")

MANIFEST = "manifests/rgb_manifest.csv"
MODALITY = "RGB"
BATCH_SIZE = 256
EPOCHS = 70
LR = 1e-3
MIN_EPOCHS = 50
PATIENCE = 10
BASE_ARCH = "C"

# Recreated loaders in 2C/2D are most stable on Windows with no worker subprocesses.
NUM_WORKERS = 0 if os.name == "nt" else 12
PREFETCH_FACTOR = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
FIGURES_DIR = Path("figures")
LOCK_PATH = CHECKPOINT_DIR / "run_part2_rgb.lock"
RESULTS_PATH = FIGURES_DIR / "part2_rgb_results.json"


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock() -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    if LOCK_PATH.exists():
        try:
            existing_pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
        except Exception:
            existing_pid = -1
        if _pid_exists(existing_pid):
            raise RuntimeError(f"Another Part 2 RGB run is already active (PID {existing_pid}).")
        LOCK_PATH.unlink(missing_ok=True)
    LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
    atexit.register(lambda: LOCK_PATH.unlink(missing_ok=True))


def evaluate_checkpoint(checkpoint_name: str, dropout: float = 0.0) -> dict:
    model, count = build_cnn(BASE_ARCH, dropout=dropout)
    ckpt_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    train_loss, train_acc, train_f1 = evaluate_model(model, train_loader, criterion, DEVICE, use_amp=True)
    val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, DEVICE, use_amp=True)
    del model
    torch.cuda.empty_cache()
    return {
        "param_count": count,
        "best_epoch": "loaded",
        "best_train_acc": train_acc,
        "best_val_acc": val_acc,
        "best_train_f1": train_f1,
        "best_val_f1": val_f1,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }


def train_or_load_l2(weight_decay: float) -> dict:
    checkpoint_name = f"part2_rgb_l2_{weight_decay}"
    ckpt_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"
    if ckpt_path.exists():
        print(f"  Loading existing checkpoint: {ckpt_path.name}")
        result = evaluate_checkpoint(checkpoint_name)
    else:
        model, count = build_cnn(BASE_ARCH)
        history = train_model(
            model, train_loader, val_loader,
            epochs=EPOCHS, lr=LR, weight_decay=weight_decay,
            patience=PATIENCE, min_epochs=MIN_EPOCHS,
            checkpoint_name=checkpoint_name,
            use_amp=True,
        )
        result = {
            "param_count": count,
            "best_epoch": history["best_epoch"],
            "best_train_acc": history["best_train_acc"],
            "best_val_acc": history["best_val_acc"],
            "best_train_f1": history["best_train_f1"],
            "best_val_f1": history["best_val_f1"],
        }
        del model
        torch.cuda.empty_cache()

    return {
        "label": f"L2={weight_decay}",
        "param_count": result["param_count"],
        "best_epoch": result["best_epoch"],
        "best_train_acc": result["best_train_acc"],
        "best_val_acc": result["best_val_acc"],
        "best_train_f1": result["best_train_f1"],
        "best_val_f1": result["best_val_f1"],
    }


def train_or_load_dropout(dropout: float) -> dict:
    checkpoint_name = f"part2_rgb_drop_{dropout}"
    ckpt_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"
    if ckpt_path.exists():
        print(f"  Loading existing checkpoint: {ckpt_path.name}")
        result = evaluate_checkpoint(checkpoint_name, dropout=dropout)
    else:
        model, count = build_cnn(BASE_ARCH, dropout=dropout)
        history = train_model(
            model, train_loader, val_loader,
            epochs=EPOCHS, lr=LR,
            patience=PATIENCE, min_epochs=MIN_EPOCHS,
            checkpoint_name=checkpoint_name,
            use_amp=True,
        )
        result = {
            "param_count": count,
            "best_epoch": history["best_epoch"],
            "best_train_acc": history["best_train_acc"],
            "best_val_acc": history["best_val_acc"],
            "best_train_f1": history["best_train_f1"],
            "best_val_f1": history["best_val_f1"],
        }
        del model
        torch.cuda.empty_cache()

    return {
        "label": f"Drop={dropout}",
        "param_count": result["param_count"],
        "best_epoch": result["best_epoch"],
        "best_train_acc": result["best_train_acc"],
        "best_val_acc": result["best_val_acc"],
        "best_train_f1": result["best_train_f1"],
        "best_val_f1": result["best_val_f1"],
    }


def train_or_load_aug(level: int) -> dict:
    checkpoint_name = f"part2_rgb_aug_{level}"
    ckpt_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"
    label = f"Aug L{level}"
    train_transform = get_augmentation(level)
    val_transform = get_augmentation(0)
    aug_train_loader, aug_val_loader, _, _ = get_dataloaders(
        MANIFEST,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=False,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    if ckpt_path.exists():
        print(f"  Loading existing checkpoint: {ckpt_path.name}")
        model, count = build_cnn(BASE_ARCH)
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
        model = model.to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss()
        _, train_acc, train_f1 = evaluate_model(model, aug_train_loader, criterion, DEVICE, use_amp=True)
        _, val_acc, val_f1 = evaluate_model(model, aug_val_loader, criterion, DEVICE, use_amp=True)
        result = {
            "param_count": count,
            "best_epoch": "loaded",
            "best_train_acc": train_acc,
            "best_val_acc": val_acc,
            "best_train_f1": train_f1,
            "best_val_f1": val_f1,
        }
        del model
    else:
        model, count = build_cnn(BASE_ARCH)
        history = train_model(
            model, aug_train_loader, aug_val_loader,
            epochs=EPOCHS, lr=LR,
            patience=PATIENCE, min_epochs=MIN_EPOCHS,
            checkpoint_name=checkpoint_name,
            use_amp=True,
        )
        result = {
            "param_count": count,
            "best_epoch": history["best_epoch"],
            "best_train_acc": history["best_train_acc"],
            "best_val_acc": history["best_val_acc"],
            "best_train_f1": history["best_train_f1"],
            "best_val_f1": history["best_val_f1"],
        }
        del model

    del aug_train_loader, aug_val_loader
    torch.cuda.empty_cache()
    return {"label": label, **result}


def train_or_load_combined(index: int, config: dict) -> dict:
    checkpoint_name = f"part2_rgb_combined_{index}"
    ckpt_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"
    train_transform = get_augmentation(config["aug"])
    val_transform = get_augmentation(0)
    c_train_loader, c_val_loader, _, _ = get_dataloaders(
        MANIFEST,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=False,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    if ckpt_path.exists():
        print(f"  Loading existing checkpoint: {ckpt_path.name}")
        model, count = build_cnn(BASE_ARCH, dropout=config["dp"])
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
        model = model.to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss()
        _, train_acc, train_f1 = evaluate_model(model, c_train_loader, criterion, DEVICE, use_amp=True)
        _, val_acc, val_f1 = evaluate_model(model, c_val_loader, criterion, DEVICE, use_amp=True)
        result = {
            "param_count": count,
            "best_epoch": "loaded",
            "best_train_acc": train_acc,
            "best_val_acc": val_acc,
            "best_train_f1": train_f1,
            "best_val_f1": val_f1,
        }
        del model
    else:
        model, count = build_cnn(BASE_ARCH, dropout=config["dp"])
        history = train_model(
            model, c_train_loader, c_val_loader,
            epochs=EPOCHS, lr=LR, weight_decay=config["wd"],
            patience=PATIENCE, min_epochs=MIN_EPOCHS,
            checkpoint_name=checkpoint_name,
            use_amp=True,
        )
        result = {
            "param_count": count,
            "best_epoch": history["best_epoch"],
            "best_train_acc": history["best_train_acc"],
            "best_val_acc": history["best_val_acc"],
            "best_train_f1": history["best_train_f1"],
            "best_val_f1": history["best_val_f1"],
        }
        del model

    del c_train_loader, c_val_loader
    torch.cuda.empty_cache()
    return {"label": config["label"], **result}


def save_summary(data: dict) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved summary: {RESULTS_PATH}")


if __name__ == "__main__":
    acquire_lock()
    start = time.time()
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Base architecture: {BASE_ARCH}")
    print(f"Workers: {NUM_WORKERS} | Prefetch: {PREFETCH_FACTOR}")
    print()

    train_loader, val_loader, train_df, val_df = get_dataloaders(
        MANIFEST,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=False,
    )
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

    # 2A
    print(f"\n{'=' * 60}\n2A: L2 Regularization\n{'=' * 60}")
    l2_values = [1e-5, 1e-4, 1e-3, 1e-2]
    l2_results = [train_or_load_l2(wd) for wd in l2_values]
    print_results_table(l2_results)
    plot_4panel(
        l2_results, l2_values,
        x_label="L2 Regularization (weight decay)",
        title=f"Part 2A - {MODALITY}: L2 Regularization",
        x_log=True,
        save_name=f"part2a_{MODALITY.lower()}_l2.png",
    )

    # 2B
    print(f"\n{'=' * 60}\n2B: Dropout\n{'=' * 60}")
    dropout_values = [0.1, 0.25, 0.4, 0.6]
    dropout_results = [train_or_load_dropout(dp) for dp in dropout_values]
    print_results_table(dropout_results)
    plot_4panel(
        dropout_results, dropout_values,
        x_label="Dropout Rate",
        title=f"Part 2B - {MODALITY}: Dropout",
        save_name=f"part2b_{MODALITY.lower()}_dropout.png",
    )

    best_l2 = max(l2_results, key=lambda r: r["best_val_f1"])["label"].split("=")[1]
    best_l2 = float(best_l2)
    best_dp = float(max(dropout_results, key=lambda r: r["best_val_f1"])["label"].split("=")[1])
    print(f"\nBest L2 from 2A: {best_l2}")
    print(f"Best Dropout from 2B: {best_dp}")

    # 2C
    print(f"\n{'=' * 60}\n2C: Data Augmentation\n{'=' * 60}")
    aug_levels = [1, 2, 3, 4]
    aug_results = [train_or_load_aug(level) for level in aug_levels]
    print_results_table(aug_results)
    plot_4panel(
        aug_results, aug_levels,
        x_label="Augmentation Level",
        title=f"Part 2C - {MODALITY}: Data Augmentation",
        save_name=f"part2c_{MODALITY.lower()}_augmentation.png",
    )
    best_aug_label = max(aug_results, key=lambda r: r["best_val_f1"])["label"]
    best_aug = int(best_aug_label.split("L")[1])
    print(f"Best Augmentation Level from 2C: {best_aug}")

    # 2D
    print(f"\n{'=' * 60}\n2D: Combined\n{'=' * 60}")
    combined_configs = [
        {"label": f"L2={best_l2}+Drop={best_dp}", "wd": best_l2, "dp": best_dp, "aug": 0},
        {"label": f"L2={best_l2}+Aug={best_aug}", "wd": best_l2, "dp": 0.0, "aug": best_aug},
        {"label": f"Drop={best_dp}+Aug={best_aug}", "wd": 0.0, "dp": best_dp, "aug": best_aug},
        {"label": "L2+Drop+Aug (all best)", "wd": best_l2, "dp": best_dp, "aug": best_aug},
    ]
    combined_results = [train_or_load_combined(i, cfg) for i, cfg in enumerate(combined_configs)]
    print_results_table(combined_results)
    plot_4panel(
        combined_results, list(range(1, 5)),
        x_label="Combined Configuration",
        title=f"Part 2D - {MODALITY}: Combined Regularization + Augmentation",
        save_name=f"part2d_{MODALITY.lower()}_combined.png",
    )

    summary = {
        "base_arch": BASE_ARCH,
        "best_l2": best_l2,
        "best_dropout": best_dp,
        "best_augmentation": best_aug,
        "l2_results": l2_results,
        "dropout_results": dropout_results,
        "augmentation_results": aug_results,
        "combined_results": combined_results,
        "duration_minutes": (time.time() - start) / 60.0,
    }
    save_summary(summary)
    print(f"\nTotal time: {summary['duration_minutes']:.1f} min")
