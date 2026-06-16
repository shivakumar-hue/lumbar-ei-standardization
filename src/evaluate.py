"""
evaluate.py
-----------
Segmentation evaluation: Dice, IoU, Precision, Recall, F1 on the
validation set. Prints per-side and per-mask-type breakdowns.

Usage:
    python src/evaluate.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch

from dataset import build_data_pairs, apply_crop
from model   import build_model

BASE_DIR   = os.path.expanduser("~/LUMINOUS_Database")
IMAGE_DIR  = os.path.join(BASE_DIR, "B-mode")
MASK_DIR   = os.path.join(BASE_DIR, "Masks")
OUTPUT_DIR = os.path.expanduser("~/unet_improved_final")
IMG_SIZE   = 256
BATCH_SIZE = 8
SEED       = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(val_pairs: list, model, out_dir: str):
    model.eval()
    all_dice, all_iou, all_prec, all_rec = [], [], [], []
    side_dice = {"Left": [], "Right": [], "Center": []}
    type_dice = {"single": [], "dual": []}
    sm = 1e-6

    with torch.no_grad():
        for bi in tqdm(range(0, len(val_pairs), BATCH_SIZE), "Evaluating"):
            batch  = val_pairs[bi : bi + BATCH_SIZE]
            imgs_l, masks_l = [], []

            for p in batch:
                img  = Image.open(p["img_path"]).convert("L")
                mask = Image.open(p["mask_path"]).convert("L")
                img, mask = apply_crop(img, mask, p["crop_type"])
                imgs_l.append(
                    torch.from_numpy(np.array(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)))
                    .float().unsqueeze(0) / 255.0
                )
                masks_l.append(
                    (torch.from_numpy(np.array(mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)))
                     .float().unsqueeze(0) / 255.0 > 0.5).float()
                )

            imgs_t  = torch.stack(imgs_l).to(device)
            masks_t = torch.stack(masks_l).to(device)
            preds   = (torch.sigmoid(model(imgs_t)) > 0.5).float()

            for i, p in enumerate(batch):
                pp = preds[i].flatten(); tt = masks_t[i].flatten()
                tp = (pp * tt).sum().item()
                fp = (pp * (1 - tt)).sum().item()
                fn = ((1 - pp) * tt).sum().item()
                d  = (2 * tp + sm) / (2 * tp + fp + fn + sm)

                all_dice.append(d)
                all_iou.append((tp + sm) / (tp + fp + fn + sm))
                all_prec.append((tp + sm) / (tp + fp + sm))
                all_rec.append((tp + sm)  / (tp + fn + sm))
                if p["side"] in side_dice:
                    side_dice[p["side"]].append(d)
                type_dice[p["mask_type"]].append(d)

    md, mi = np.mean(all_dice), np.mean(all_iou)
    mp, mr = np.mean(all_prec), np.mean(all_rec)
    mf     = 2 * mp * mr / (mp + mr + 1e-6)

    print("=" * 55)
    print("  SEGMENTATION RESULTS (Midpoint-Crop)")
    print("=" * 55)
    print(f"  Dice      : {md:.4f}")
    print(f"  IoU       : {mi:.4f}")
    print(f"  Precision : {mp:.4f}")
    print(f"  Recall    : {mr:.4f}")
    print(f"  F1        : {mf:.4f}")
    print()
    for s, dices in side_dice.items():
        if dices:
            print(f"  {s:6s} Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}  (n={len(dices)})")
    for t, dices in type_dice.items():
        if dices:
            print(f"  {t:6s} Dice: {np.mean(dices):.4f}  (n={len(dices)})")
    print("=" * 55)

    # ── Histogram plots ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, sc, nm, cl in zip(
        axes,
        [all_dice, all_iou, all_prec, all_rec],
        ["Dice", "IoU", "Precision", "Recall"],
        ["steelblue", "teal", "coral", "mediumpurple"],
    ):
        ax.hist(sc, bins=20, color=cl, edgecolor="k", alpha=0.8)
        ax.axvline(np.mean(sc), color="red", linestyle="--",
                   label=f"Mean={np.mean(sc):.3f}")
        ax.set_title(nm); ax.legend()
    plt.suptitle("Segmentation Performance — Midpoint-Crop Strategy", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "segmentation_scores.png"), dpi=100)
    plt.close()
    print("Saved: segmentation_scores.png")

    return {"dice": md, "iou": mi, "precision": mp, "recall": mr, "f1": mf}


if __name__ == "__main__":
    import random
    data_pairs = build_data_pairs(IMAGE_DIR, MASK_DIR)
    random.seed(SEED); random.shuffle(data_pairs)
    sp        = int(0.8 * len(data_pairs))
    val_pairs = data_pairs[sp:]

    model = build_model(device)
    model.load_state_dict(torch.load(
        os.path.join(OUTPUT_DIR, "unet_best.pth"), map_location=device))

    evaluate(val_pairs, model, OUTPUT_DIR)
