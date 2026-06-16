"""
train.py
--------
Training script for the Lumbar Multifidus U-Net segmentation model.

Usage:
    python src/train.py

Outputs (saved to OUTPUT_DIR):
    unet_best.pth           -- best model weights (by val loss)
    training_curves.png     -- loss and Dice plots
    training_history.csv    -- epoch-level metrics
"""

import os
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from dataset import build_data_pairs, get_loaders
from model   import build_model
from loss    import CombinedLoss

# ── Configuration ──────────────────────────────────────────────────────────
SEED         = 42
BASE_DIR     = os.path.expanduser("~/LUMINOUS_Database")
IMAGE_DIR    = os.path.join(BASE_DIR, "B-mode")
MASK_DIR     = os.path.join(BASE_DIR, "Masks")
OUTPUT_DIR   = os.path.expanduser("~/unet_improved_final")
IMG_SIZE     = 256
BATCH_SIZE   = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS   = 50
PATIENCE     = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Batch Dice (for monitoring) ────────────────────────────────────────────
def dice_batch(logits, targets, threshold=0.5, smooth=1e-6):
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return ((2 * inter + smooth) / (union + smooth)).mean().item()


# ── Single epoch ───────────────────────────────────────────────────────────
def run_epoch(model, loader, loss_fn, optimizer, scaler, train=True):
    model.train() if train else model.eval()
    total_loss, total_dice = 0.0, 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                loss  = loss_fn(preds, masks)
            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item()
            total_dice += dice_batch(preds.detach() if train else preds, masks)

    n = len(loader)
    return total_loss / n, total_dice / n


# ── Main training loop ─────────────────────────────────────────────────────
def train():
    data_pairs = build_data_pairs(IMAGE_DIR, MASK_DIR)
    train_loader, val_loader, _, (_, _) = get_loaders(
        data_pairs, batch_size=BATCH_SIZE, seed=SEED
    )

    model     = build_model(device)
    loss_fn   = CombinedLoss(weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5, min_lr=1e-6
    )
    scaler    = torch.cuda.amp.GradScaler()

    model_path = os.path.join(OUTPUT_DIR, "unet_best.pth")
    train_losses, val_losses   = [], []
    train_dices,  val_dices    = [], []
    best_val, no_improve       = float("inf"), 0

    print(f"Training up to {NUM_EPOCHS} epochs (early stop patience={PATIENCE})")
    print(f"{'Ep':>4} {'TrLoss':>8} {'TrDice':>8} {'VaLoss':>8} {'VaDice':>8} {'LR':>10}")
    print("-" * 55)

    for ep in range(1, NUM_EPOCHS + 1):
        tl, td = run_epoch(model, train_loader, loss_fn, optimizer, scaler, train=True)
        vl, vd = run_epoch(model, val_loader,   loss_fn, optimizer, scaler, train=False)

        train_losses.append(tl); val_losses.append(vl)
        train_dices.append(td);  val_dices.append(vd)
        scheduler.step(vl)
        lr = optimizer.param_groups[0]["lr"]
        print(f"{ep:>4} {tl:>8.4f} {td:>8.4f} {vl:>8.4f} {vd:>8.4f} {lr:>10.2e}")

        if vl < best_val:
            best_val, no_improve = vl, 0
            torch.save(model.state_dict(), model_path)
            print(f"     *** Saved (dice={vd:.4f}) ***")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stop at epoch {ep}")
                break

    # ── Save training curves ────────────────────────────────────────────────
    n_ep = len(train_losses)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, tr, va, title in zip(
        axes,
        [train_losses, train_dices],
        [val_losses,   val_dices],
        ["Combined Loss", "Dice Score"],
    ):
        ax.plot(range(1, n_ep + 1), tr, "b-o", ms=3, label="Train")
        ax.plot(range(1, n_ep + 1), va, "r-o", ms=3, label="Val")
        ax.set_title(title); ax.legend(); ax.grid(True)
    plt.suptitle("U-Net Training — Midpoint-Crop Strategy", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=100)
    plt.close()

    # ── Save history CSV ────────────────────────────────────────────────────
    pd.DataFrame({
        "epoch":      list(range(1, n_ep + 1)),
        "train_loss": train_losses,
        "val_loss":   val_losses,
        "train_dice": train_dices,
        "val_dice":   val_dices,
    }).to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)

    print(f"\nBest val loss : {best_val:.4f}")
    print(f"Model saved   : {model_path}")
    print("Training curves: training_curves.png")


if __name__ == "__main__":
    train()
