"""
dataset.py
----------
LuminousDataset class with verified Midpoint-Crop strategy for the
LUMINOUS B-mode ultrasound database.

Key insight: 45 of 341 images show BOTH lumbar multifidus (LM) muscles.
    - Mask1.tif = RIGHT LM  (centroid x > 410)
    - Mask2.tif = LEFT LM   (centroid x < 410)
    - _Mask.tif = single muscle, use full image

Minimum inter-muscle gap verified at 20px; midpoint x=410 always valid.
"""

import os
import re
import random
from collections import Counter

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader


IMG_SIZE = 256


# ---------------------------------------------------------------------------
# Helper: determine side for single-mask images from centroid position
# ---------------------------------------------------------------------------
def get_mask_side_single(mask_path: str) -> str:
    """For single masks: determine LM side from mask centroid x-position."""
    m = np.array(Image.open(mask_path).convert("L"))
    cols = np.where(m > 127)[1]
    if len(cols) == 0:
        return "Unknown"
    cx = cols.mean()
    W = m.shape[1]
    if cx < W * 0.45:
        return "Left"
    elif cx > W * 0.55:
        return "Right"
    else:
        return "Center"


# ---------------------------------------------------------------------------
# Midpoint-crop: isolate a single LM from dual-muscle images
# ---------------------------------------------------------------------------
def apply_crop(img_pil: Image.Image, mask_pil: Image.Image, crop_type: str):
    """
    Apply midpoint crop based on which LM muscle we want.

    Args:
        img_pil   : PIL image (grayscale)
        mask_pil  : PIL mask  (grayscale)
        crop_type : 'full' | 'left_half' | 'right_half'

    Returns:
        (cropped_img, cropped_mask) as PIL images
    """
    if crop_type == "full":
        return img_pil, mask_pil

    W, H = img_pil.size  # PIL: (width, height)
    mid = W // 2         # x = 410 for 820-px wide LUMINOUS images

    if crop_type == "left_half":
        box = (0, 0, mid, H)
    else:  # right_half
        box = (mid, 0, W, H)

    return img_pil.crop(box), mask_pil.crop(box)


# ---------------------------------------------------------------------------
# Build dataset pairs from directory
# ---------------------------------------------------------------------------
def build_data_pairs(image_dir: str, mask_dir: str) -> list:
    """
    Scan the LUMINOUS directory structure and build a list of
    (image, mask, metadata) dicts with crop assignments.

    Expected naming:
        images : {subject_id}_{visit_id}_*.tif
        masks  : {subject_id}_{visit_id}_*Mask.tif
                 {subject_id}_{visit_id}_*Mask1.tif  (RIGHT LM)
                 {subject_id}_{visit_id}_*Mask2.tif  (LEFT  LM)
    """
    image_files = sorted(f for f in os.listdir(image_dir) if f.endswith(".tif"))
    mask_files  = sorted(f for f in os.listdir(mask_dir)  if f.endswith(".tif"))

    data_pairs = []
    for img_file in image_files:
        m = re.match(r"^(\d+)_(\d+)_", img_file)
        if not m:
            continue
        sid, vid = m.group(1), m.group(2)
        masks = sorted(f for f in mask_files if f.startswith(f"{sid}_{vid}_"))

        for mf in masks:
            mask_path = os.path.join(mask_dir,  mf)
            img_path  = os.path.join(image_dir, img_file)

            if "Mask1" in mf:
                side, crop_type = "Right", "right_half"
            elif "Mask2" in mf:
                side, crop_type = "Left", "left_half"
            else:
                side      = get_mask_side_single(mask_path)
                crop_type = "full"

            data_pairs.append({
                "img_path"  : img_path,
                "mask_path" : mask_path,
                "subject_id": sid,
                "visit_id"  : vid,
                "visit_int" : int(vid),
                "side"      : side,
                "crop_type" : crop_type,
                "mask_type" : "dual" if crop_type != "full" else "single",
                "base"      : f"{sid}_{vid}",
            })

    sides = Counter(p["side"]      for p in data_pairs)
    crops = Counter(p["crop_type"] for p in data_pairs)
    print(f"Total pairs      : {len(data_pairs)}")
    print(f"  Left LM        : {sides['Left']}")
    print(f"  Right LM       : {sides['Right']}")
    print(f"  Center LM      : {sides['Center']}")
    print(f"  Full image     : {crops['full']}  (single mask)")
    print(f"  Left half crop : {crops['left_half']}  (Mask2 — LEFT LM)")
    print(f"  Right half crop: {crops['right_half']}  (Mask1 — RIGHT LM)")
    print(f"  Subjects       : {len(set(p['subject_id'] for p in data_pairs))}")
    return data_pairs


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class LuminousDataset(Dataset):
    """
    PyTorch Dataset for LUMINOUS B-mode ultrasound images.

    Applies midpoint-crop strategy to handle dual-muscle images correctly,
    then optionally augments (training only) before returning tensors.
    """

    def __init__(self, pairs: list, size: int = IMG_SIZE, augment: bool = False):
        self.pairs   = pairs
        self.size    = size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p    = self.pairs[idx]
        img  = Image.open(p["img_path"]).convert("L")
        mask = Image.open(p["mask_path"]).convert("L")

        # Isolate single LM via midpoint crop
        img, mask = apply_crop(img, mask, p["crop_type"])

        img  = img.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                img  = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                img  = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            ang  = random.uniform(-15, 15)
            img  = img.rotate(ang, resample=Image.BILINEAR)
            mask = mask.rotate(ang, resample=Image.NEAREST)
            if random.random() > 0.5:
                img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))

        img_t  = torch.from_numpy(np.array(img)).float().unsqueeze(0)  / 255.0
        mask_t = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0
        return img_t, (mask_t > 0.5).float()


# ---------------------------------------------------------------------------
# Convenience: build train/val/full DataLoaders
# ---------------------------------------------------------------------------
def get_loaders(data_pairs: list,
                batch_size: int = 8,
                val_split:  float = 0.2,
                num_workers: int = 2,
                seed: int = 42):
    """
    Split data_pairs into train/val, return DataLoaders.

    Returns:
        train_loader, val_loader, full_loader, (train_pairs, val_pairs)
    """
    random.seed(seed)
    pairs = data_pairs.copy()
    random.shuffle(pairs)

    sp          = int((1 - val_split) * len(pairs))
    train_pairs = pairs[:sp]
    val_pairs   = pairs[sp:]

    train_ds = LuminousDataset(train_pairs, augment=True)
    val_ds   = LuminousDataset(val_pairs,   augment=False)
    full_ds  = LuminousDataset(pairs,       augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    full_loader  = DataLoader(full_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")
    return train_loader, val_loader, full_loader, (train_pairs, val_pairs)
