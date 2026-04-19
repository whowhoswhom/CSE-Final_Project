"""
training.py - Training loop, dataset, evaluation, and metrics.

Features:
  - Early stopping eligible only after epoch 50 (patience=10)
  - ReduceLROnPlateau scheduler
  - Per-epoch accuracy + weighted F1 tracking
  - Best-checkpoint saving & restoring
  - Manifest-driven PyTorch Dataset
"""

import os
import copy
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from pathlib import Path
from typing import Any

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _running_in_notebook() -> bool:
    """Best-effort detection for notebook kernels on Windows."""
    return "ipykernel" in sys.modules


def recommended_num_workers(cap: int = 24) -> int:
    """
    Choose an aggressive but practical worker count for patch datasets.
    Leaves a few logical CPUs for the main process and OS.
    """
    logical = os.cpu_count() or 8
    return max(4, min(cap, logical - 4))


def _loader_worker_init(_: int) -> None:
    """Avoid per-worker thread oversubscription when workers spin up."""
    torch.set_num_threads(1)


def _autocast_args(device: torch.device, use_amp: bool) -> tuple[str, torch.dtype, bool]:
    """Return safe autocast settings for the current device."""
    if device.type == "cuda":
        return "cuda", torch.float16, use_amp
    return "cpu", torch.bfloat16, False


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class PatchDataset(Dataset):
    """Loads image patches from a manifest CSV."""

    def __init__(self, manifest_df: pd.DataFrame, transform: transforms.Compose | None = None, img_size: int = 64) -> None:
        self.df = manifest_df.reset_index(drop=True)
        self.img_size = img_size
        self.paths = self.df["patch_path"].tolist()
        self.labels = self.df["label_id"].astype(int).tolist()
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        with Image.open(self.paths[idx]) as src:
            img = src.convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


def get_dataloaders(manifest_path: str | Path, batch_size: int = 256, img_size: int = 64,
                    train_transform: transforms.Compose | None = None,
                    val_transform: transforms.Compose | None = None,
                    num_workers: int | None = None,
                    prefetch_factor: int = 4,
                    persistent_workers: bool | None = None) -> tuple[DataLoader[Any], DataLoader[Any], pd.DataFrame, pd.DataFrame]:
    """
    Create train and val DataLoaders from a manifest CSV.
    Returns (train_loader, val_loader, train_df, val_df).
    """
    df = pd.read_csv(manifest_path)
    train_df = pd.DataFrame(df[df["split"] == "train"])
    val_df = pd.DataFrame(df[df["split"] == "val"])

    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    if train_transform is None:
        train_transform = val_transform

    if num_workers is None:
        num_workers = recommended_num_workers()

    if os.name == "nt" and num_workers > 0 and not _running_in_notebook():
        print(
            "Windows script runner detected. Multi-worker loading is allowed only "
            "from guarded entrypoints; if you hit duplicate launches, rerun with "
            "num_workers=0 or a notebook kernel."
        )

    train_ds = PatchDataset(train_df, transform=train_transform, img_size=img_size)
    val_ds = PatchDataset(val_df, transform=val_transform, img_size=img_size)

    pin_memory = DEVICE.type == "cuda"
    # Notebook experiments frequently rebuild loaders inside loops. On Windows,
    # persistent workers can linger across those re-creations and exhaust
    # resources, so default them off in notebook kernels unless explicitly set.
    if persistent_workers is None:
        use_persistent_workers = num_workers > 0 and not (_running_in_notebook() and os.name == "nt")
    else:
        use_persistent_workers = persistent_workers
    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = use_persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["worker_init_fn"] = _loader_worker_init

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            **loader_kwargs)

    return train_loader, val_loader, train_df, val_df


# ═══════════════════════════════════════════════════════════════════════════
# Augmentation presets (safe transforms only - no random crop)
# ═══════════════════════════════════════════════════════════════════════════

def get_augmentation(level: int, img_size: int = 64) -> transforms.Compose:
    """
    Return a transform pipeline for the given augmentation level.
    Level 0: no augmentation (resize + toTensor)
    Level 1: horizontal flip
    Level 2: flip + rotation(10) + affine translate
    Level 3: flip + rotation(20) + affine + color jitter
    Level 4: flip + rotation(30) + affine + stronger color jitter
    """
    base: list[Any] = [transforms.Resize((img_size, img_size))]

    if level == 0:
        base.append(transforms.ToTensor())
        return transforms.Compose(base)

    if level >= 1:
        base.append(transforms.RandomHorizontalFlip())
    if level >= 2:
        base.append(transforms.RandomRotation(10))
        base.append(transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)))
    if level >= 3:
        base[-1] = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        base[-2] = transforms.RandomRotation(20)
        base.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    if level >= 4:
        # Replace previous rotation/affine with stronger versions
        for i, t in enumerate(base):
            if isinstance(t, transforms.RandomRotation):
                base[i] = transforms.RandomRotation(30)
            if isinstance(t, transforms.RandomAffine):
                base[i] = transforms.RandomAffine(degrees=0, translate=(0.15, 0.15))
            if isinstance(t, transforms.ColorJitter):
                base[i] = transforms.ColorJitter(brightness=0.3, contrast=0.3)

    base.append(transforms.ToTensor())
    return transforms.Compose(base)


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_model(model: nn.Module, train_loader: DataLoader[Any], val_loader: DataLoader[Any],
                epochs: int = 60, lr: float = 1e-3,
                weight_decay: float = 0.0, patience: int = 10, min_epochs: int = 50,
                checkpoint_name: str = "model", param_groups: list[dict[str, Any]] | None = None,
                device: torch.device | None = None,
                use_amp: bool | None = None) -> dict[str, Any]:
    """
    Train a model with early stopping eligible only after min_epochs.

    Args:
        model: nn.Module
        train_loader, val_loader: DataLoaders
        epochs: maximum epochs (should be > min_epochs + patience)
        lr: learning rate (ignored if param_groups is provided)
        weight_decay: L2 regularization
        patience: early stopping patience (active after min_epochs)
        min_epochs: minimum epochs before early stopping can trigger
        checkpoint_name: name prefix for saved checkpoint
        param_groups: optional list of param group dicts (for differential LR)
        device: torch device

    Returns:
        dict with keys: train_acc, val_acc, train_f1, val_f1, train_loss,
                        val_loss, best_epoch, best_val_f1, best_val_acc,
                        lr_history
    """
    if device is None:
        device = DEVICE
    model = model.to(device)
    amp_enabled = bool(use_amp) if use_amp is not None else device.type == "cuda"
    amp_device_type, amp_dtype, amp_enabled = _autocast_args(device, amp_enabled)
    scaler = GradScaler(enabled=amp_enabled)

    criterion = nn.CrossEntropyLoss()

    if param_groups is not None:
        optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    history: dict[str, Any] = {
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": [],
        "train_loss": [], "val_loss": [],
        "lr_history": [],
    }

    best_val_f1 = -1.0
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        assert train_loader.dataset is not None
        train_loss = running_loss / len(train_loader.dataset)  # type: ignore[arg-type]
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division="warn")

        # ── Validate ──
        val_loss, val_acc, val_f1 = evaluate_model(
            model, val_loader, criterion, device, use_amp=amp_enabled
        )

        # ── Record ──
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr_history"].append(current_lr)

        # ── LR scheduling (based on val F1) ──
        scheduler.step(val_f1)

        # ── Best checkpoint tracking ──
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # ── Progress ──
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"Acc: {train_acc:.4f}/{val_acc:.4f} | "
                  f"F1: {train_f1:.4f}/{val_f1:.4f} | "
                  f"LR: {current_lr:.2e}")

        # ── Early stopping (only after min_epochs) ──
        if epoch >= min_epochs and patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    # ── Restore best model ──
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, ckpt_path)

    # ── Final metrics from best checkpoint ──
    val_loss_best, val_acc_best, val_f1_best = evaluate_model(
        model, val_loader, criterion, device, use_amp=amp_enabled
    )
    _, train_acc_best, train_f1_best = evaluate_model(
        model, train_loader, criterion, device, use_amp=amp_enabled
    )

    history["best_epoch"] = best_epoch
    history["best_val_f1"] = val_f1_best
    history["best_val_acc"] = val_acc_best
    history["best_train_acc"] = train_acc_best
    history["best_train_f1"] = train_f1_best

    print(f"  Best epoch: {best_epoch} | "
          f"Val Acc: {val_acc_best:.4f} | Val F1: {val_f1_best:.4f}")

    return history


def evaluate_model(model: nn.Module, loader: DataLoader[Any],
                   criterion: nn.Module | None = None,
                   device: torch.device | None = None,
                   use_amp: bool | None = None) -> tuple[Any, Any, Any]:
    """Evaluate model on a DataLoader. Returns (loss, accuracy, weighted_f1)."""
    if device is None:
        device = DEVICE
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    amp_enabled = bool(use_amp) if use_amp is not None else device.type == "cuda"
    amp_device_type, amp_dtype, amp_enabled = _autocast_args(device, amp_enabled)

    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    assert loader.dataset is not None
    avg_loss = running_loss / len(loader.dataset)  # type: ignore[arg-type]
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division="warn")

    return avg_loss, acc, f1


def get_predictions(model: nn.Module, loader: DataLoader[Any],
                    device: torch.device | None = None) -> tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
    """Get all predictions and true labels from a DataLoader."""
    if device is None:
        device = DEVICE
    model.eval()
    amp_enabled = device.type == "cuda"
    amp_device_type, amp_dtype, amp_enabled = _autocast_args(device, amp_enabled)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)
