"""
biomarkers.py
-------------
Extract Echo Intensity (EI) and Cross-Sectional Area (CSA) from U-Net
predicted masks, then apply all four standardization methods and run
Heckmatt grading, left-vs-right asymmetry, Bland-Altman, and longitudinal
reproducibility analysis.

Usage:
    python src/biomarkers.py

Outputs (saved to OUTPUT_DIR):
    echointensity_results.csv
    subject_stats.csv
    side_asymmetry.csv
    standardization_comparison.png
    heckmatt_grading.png
    side_analysis.png
    bland_altman.png
    longitudinal_analysis.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy.stats import mannwhitneyu
import torch

from dataset import build_data_pairs, apply_crop
from model   import build_model

# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.expanduser("~/LUMINOUS_Database")
IMAGE_DIR  = os.path.join(BASE_DIR, "B-mode")
MASK_DIR   = os.path.join(BASE_DIR, "Masks")
OUTPUT_DIR = os.path.expanduser("~/unet_improved_final")
IMG_SIZE   = 256
BATCH_SIZE = 8

# LUMINOUS physical scale: 820×614 px = 12 cm × 6 cm
CM_PER_PX_H = 6.0  / 614.0
CM_PER_PX_W = 12.0 / 820.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Step 1: Extract EI + CSA from predicted masks ─────────────────────────
def extract_biomarkers(data_pairs: list, model) -> pd.DataFrame:
    """
    Run the U-Net on all images and compute per-sample EI and CSA.

    CSA is computed in cm² using the physical pixel scale, corrected
    for the midpoint-crop (half-width images use half the x-scale).
    """
    model.eval()
    records = []

    with torch.no_grad():
        for bi in tqdm(range(0, len(data_pairs), BATCH_SIZE), "Extracting EI+CSA"):
            batch = data_pairs[bi : bi + BATCH_SIZE]
            imgs_l, crop_dims = [], []

            for p in batch:
                img  = Image.open(p["img_path"]).convert("L")
                mask = Image.open(p["mask_path"]).convert("L")
                img, mask = apply_crop(img, mask, p["crop_type"])
                orig_w, orig_h = img.size
                crop_dims.append((orig_h, orig_w))
                t = np.array(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
                imgs_l.append(torch.from_numpy(t).float().unsqueeze(0) / 255.0)

            imgs_t = torch.stack(imgs_l).to(device)
            preds  = (torch.sigmoid(model(imgs_t)) > 0.5).float()

            for i, p in enumerate(batch):
                it = imgs_t[i].squeeze()
                mt = preds[i].squeeze()
                ms = mt.sum().item()

                orig_h, orig_w = crop_dims[i]
                px_h = orig_h / IMG_SIZE
                px_w = orig_w / IMG_SIZE

                if ms > 0:
                    roi_pixels = it[mt > 0.5].cpu().numpy()
                    ei  = float(roi_pixels.mean())
                    eis = float(roi_pixels.std())
                    csa = float(ms * px_h * CM_PER_PX_H * px_w * CM_PER_PX_W)
                    mr  = ms / mt.numel()
                else:
                    all_pixels = it.cpu().numpy().flatten()
                    ei, eis, csa, mr = float(all_pixels.mean()), float(all_pixels.std()), 0.0, 0.0

                records.append({
                    "subject_id": p["subject_id"],
                    "visit_id"  : p["visit_id"],
                    "visit_int" : p["visit_int"],
                    "side"      : p["side"],
                    "crop_type" : p["crop_type"],
                    "mask_type" : p["mask_type"],
                    "base"      : p["base"],
                    "ei_raw"    : ei,
                    "ei_std"    : eis,
                    "csa_cm2"   : csa,
                    "mask_ratio": float(mr),
                })

    df = pd.DataFrame(records)
    print(f"Extracted: {len(df)} samples")
    print(f"  Left: {len(df[df['side']=='Left'])}  Right: {len(df[df['side']=='Right'])}  "
          f"Center: {len(df[df['side']=='Center'])}")
    print(f"Mean EI = {df['ei_raw'].mean():.4f}   Mean CSA = {df['csa_cm2'].mean():.2f} cm²")
    return df


# ── Step 2: Standardization ────────────────────────────────────────────────
def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Add four standardized EI columns to the DataFrame."""
    ei = df["ei_raw"].values
    df["ei_zscore"]  = (ei - ei.mean()) / (ei.std() + 1e-8)
    df["ei_minmax"]  = (ei - ei.min())  / (ei.max() - ei.min() + 1e-8)
    df["ei_phantom"] = (ei / ei.mean()) * 100.0
    q25, q75 = np.percentile(ei, 25), np.percentile(ei, 75)
    df["ei_robust"]  = (ei - np.median(ei)) / (q75 - q25 + 1e-8)

    cv_raw     = (ei.std()                        / ei.mean()) * 100
    cv_phantom = (df["ei_phantom"].std()           / df["ei_phantom"].mean()) * 100
    print(f"CV Raw: {cv_raw:.2f}%  |  CV Phantom: {cv_phantom:.2f}%")
    return df


def plot_standardization(df: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, (title, col, color) in zip(axes, [
        ("Raw EI (normalized pixel)",      "ei_raw",    "steelblue"),
        ("Z-score Standardized",           "ei_zscore", "teal"),
        ("Min-Max [0,1]",                  "ei_minmax", "coral"),
        ("Phantom-Reference (% of mean)",  "ei_phantom","mediumpurple"),
    ]):
        v = df[col].values
        ax.hist(v, bins=30, color=color, edgecolor="k", alpha=0.8)
        ax.axvline(v.mean(), color="red", linestyle="--",
                   label=f"Mean={v.mean():.3f}\nStd={v.std():.3f}")
        ax.set_title(title, fontsize=11); ax.legend()
        ax.set_xlabel("Value"); ax.set_ylabel("Count")
    plt.suptitle("EI Standardization Methods — LUMINOUS LM Muscle", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "standardization_comparison.png"), dpi=100)
    plt.close()
    print("Saved: standardization_comparison.png")


# ── Step 3: Heckmatt Grading ───────────────────────────────────────────────
def heckmatt_grade(z: float) -> int:
    """Heckmatt et al. (1982): Grade I=Normal, II=Mild, III=Moderate, IV=Severe."""
    if   z < -0.5: return 1
    elif z <  0.5: return 2
    elif z <  1.5: return 3
    else:          return 4


def apply_heckmatt(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    df["heckmatt"] = df["ei_zscore"].apply(heckmatt_grade)
    gc = df["heckmatt"].value_counts().sort_index()
    print("Heckmatt Distribution:")
    for g, c in gc.items():
        print(f"  Grade {g}: {c:4d} ({c / len(df) * 100:.1f}%)")

    colors = ["#2ECC71", "#F39C12", "#E74C3C", "#8E44AD"]
    gl     = {1: "Grade I\n(Normal)", 2: "Grade II\n(Mild)",
              3: "Grade III\n(Moderate)", 4: "Grade IV\n(Severe)"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar([gl[g] for g in sorted(gc.index)],
                [gc[g] for g in sorted(gc.index)],
                color=colors[:len(gc)], edgecolor="k")
    axes[0].set_title("Overall Heckmatt Distribution", fontsize=12)
    axes[0].set_ylabel("Count")
    sg = df.groupby(["side", "heckmatt"]).size().unstack(fill_value=0)
    sg.plot(kind="bar", ax=axes[1], color=colors[:4], edgecolor="k")
    axes[1].set_title("Heckmatt by LM Side", fontsize=12)
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="Grade", labels=[f"Grade {g}" for g in sg.columns])
    plt.suptitle("Heckmatt Grading — LM Muscle EI", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heckmatt_grading.png"), dpi=100)
    plt.close()
    print("Saved: heckmatt_grading.png")
    return df


# ── Step 4: Left vs Right Asymmetry ───────────────────────────────────────
def side_analysis(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    df_dual  = df[df["mask_type"] == "dual"].copy()
    left_ei  = df_dual[df_dual["side"] == "Left"]["ei_raw"].values
    right_ei = df_dual[df_dual["side"] == "Right"]["ei_raw"].values

    pval = 1.0
    if len(left_ei) > 1 and len(right_ei) > 1:
        _, pval = mannwhitneyu(left_ei, right_ei, alternative="two-sided")
        print(f"Left  EI: {left_ei.mean():.4f} ± {left_ei.std():.4f}")
        print(f"Right EI: {right_ei.mean():.4f} ± {right_ei.std():.4f}")
        print(f"Mann-Whitney p={pval:.4f} ({'Significant' if pval < 0.05 else 'Not significant'})")

    dl = df_dual[df_dual["side"] == "Left"][["subject_id","visit_id","ei_raw","csa_cm2"]].copy()
    dr = df_dual[df_dual["side"] == "Right"][["subject_id","visit_id","ei_raw","csa_cm2"]].copy()
    dl = dl.rename(columns={"ei_raw": "ei_l", "csa_cm2": "csa_l"})
    dr = dr.rename(columns={"ei_raw": "ei_r", "csa_cm2": "csa_r"})
    dsym = dl.merge(dr, on=["subject_id", "visit_id"])

    if len(dsym) > 0:
        dsym["ei_asym_pct"]  = (abs(dsym["ei_l"]  - dsym["ei_r"])  /
                                ((dsym["ei_l"]  + dsym["ei_r"])  / 2 + 1e-8)) * 100
        dsym["csa_asym_pct"] = (abs(dsym["csa_l"] - dsym["csa_r"]) /
                                ((dsym["csa_l"] + dsym["csa_r"]) / 2 + 1e-8)) * 100
        print(f"\nPaired records : {len(dsym)}")
        print(f"EI asymmetry   : {dsym['ei_asym_pct'].mean():.2f}% ± {dsym['ei_asym_pct'].std():.2f}%")
        print(f"CSA asymmetry  : {dsym['csa_asym_pct'].mean():.2f}%")
        dsym.to_csv(os.path.join(out_dir, "side_asymmetry.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    bp = axes[0].boxplot([left_ei, right_ei], labels=["Left LM", "Right LM"], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue"); bp["boxes"][1].set_facecolor("lightcoral")
    axes[0].set_title(f"EI: Left vs Right LM\n(p={pval:.4f})"); axes[0].set_ylabel("Raw EI")

    lc = df_dual[df_dual["side"] == "Left"]["csa_cm2"].values
    rc = df_dual[df_dual["side"] == "Right"]["csa_cm2"].values
    bp2 = axes[1].boxplot([lc, rc], labels=["Left LM", "Right LM"], patch_artist=True)
    bp2["boxes"][0].set_facecolor("lightblue"); bp2["boxes"][1].set_facecolor("lightcoral")
    axes[1].set_title("CSA (cm²): Left vs Right LM"); axes[1].set_ylabel("CSA (cm²)")

    if len(dsym) > 0:
        axes[2].hist(dsym["ei_asym_pct"], bins=15, color="steelblue", edgecolor="k", alpha=0.8)
        axes[2].axvline(dsym["ei_asym_pct"].mean(), color="red", linestyle="--",
                        label=f"Mean={dsym['ei_asym_pct'].mean():.2f}%")
        axes[2].set_title("Real EI Asymmetry (%)"); axes[2].set_xlabel("%"); axes[2].legend()

    plt.suptitle("Left vs Right LM — Dual-Mask Subjects", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "side_analysis.png"), dpi=100)
    plt.close()
    print("Saved: side_analysis.png")
    return dsym if len(dsym) > 0 else pd.DataFrame()


# ── Step 5: Bland-Altman ───────────────────────────────────────────────────
def bland_altman(df: pd.DataFrame, out_dir: str):
    m1 = df["ei_raw"].values
    m2 = df["ei_phantom"].values / 100.0
    ba_mean = (m1 + m2) / 2
    ba_diff = m1 - m2
    bd, bs  = ba_diff.mean(), ba_diff.std()
    bu, bl  = bd + 1.96 * bs, bd - 1.96 * bs

    plt.figure(figsize=(10, 6))
    plt.scatter(ba_mean, ba_diff, alpha=0.5, s=25, color="steelblue",
                edgecolors="k", linewidths=0.3)
    plt.axhline(bd, color="red",    linewidth=2, label=f"Mean diff={bd:.4f}")
    plt.axhline(bu, color="orange", linestyle="--", linewidth=1.5, label=f"+1.96SD={bu:.4f}")
    plt.axhline(bl, color="orange", linestyle="--", linewidth=1.5, label=f"-1.96SD={bl:.4f}")
    plt.fill_between([ba_mean.min(), ba_mean.max()], bl, bu, alpha=0.08, color="orange")
    plt.xlabel("Mean of Raw EI & Phantom-Std EI", fontsize=12)
    plt.ylabel("Difference (Raw - Phantom-Std)",  fontsize=12)
    plt.title("Bland-Altman Plot\nLumbar Multifidus — LUMINOUS", fontsize=13)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bland_altman.png"), dpi=100)
    plt.close()
    print(f"LoA: [{bl:.4f}, {bu:.4f}]")
    print("Saved: bland_altman.png")


# ── Step 6: Longitudinal Reproducibility ───────────────────────────────────
def longitudinal_analysis(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    ss = df.groupby("subject_id").agg(
        n_visits  = ("visit_id",  "count"),
        mean_ei   = ("ei_raw",    "mean"),
        std_ei    = ("ei_raw",    "std"),
        mean_csa  = ("csa_cm2",   "mean"),
        std_csa   = ("csa_cm2",   "std"),
    ).reset_index().fillna(0)
    ss["cv_ei"]  = (ss["std_ei"]  / (ss["mean_ei"]  + 1e-8)) * 100
    ss["cv_csa"] = (ss["std_csa"] / (ss["mean_csa"] + 1e-8)) * 100
    multi_v = ss[ss["n_visits"] > 1]
    print(f"\nMulti-visit subjects       : {len(multi_v)}")
    print(f"Mean within-subject EI CV  : {multi_v['cv_ei'].mean():.2f}%")
    print(f"Mean within-subject CSA CV : {multi_v['cv_csa'].mean():.2f}%")

    top_s = multi_v.nlargest(8, "n_visits")["subject_id"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for sid in top_s:
        sub = df[df["subject_id"] == sid].sort_values("visit_int")
        axes[0][0].plot(sub["visit_int"], sub["ei_raw"],    "o-", label=f"S{sid}", alpha=0.8)
        axes[0][1].plot(sub["visit_int"], sub["ei_zscore"], "o-", label=f"S{sid}", alpha=0.8)
        axes[1][0].plot(sub["visit_int"], sub["csa_cm2"],   "o-", label=f"S{sid}", alpha=0.8)
    for ax, title, ylabel in zip(
        [axes[0][0], axes[0][1], axes[1][0]],
        ["Raw EI Across Sessions", "Z-score EI Across Sessions", "CSA (cm²) Across Sessions"],
        ["Raw EI", "Z-score", "CSA (cm²)"],
    ):
        ax.set_title(title); ax.set_xlabel("Session ID"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, ncol=2)

    if len(multi_v) > 0:
        axes[1][1].hist(multi_v["cv_ei"].clip(0, 50), bins=20,
                        color="steelblue", edgecolor="k", alpha=0.8)
        axes[1][1].axvline(multi_v["cv_ei"].mean(), color="red", linestyle="--",
                           label=f"Mean={multi_v['cv_ei'].mean():.2f}%")
        axes[1][1].set_title("Within-Subject EI CV (Longitudinal Reproducibility)")
        axes[1][1].set_xlabel("CV (%)"); axes[1][1].legend()

    plt.suptitle("Longitudinal LM EI & CSA — LUMINOUS Database", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "longitudinal_analysis.png"), dpi=100)
    plt.close()
    print("Saved: longitudinal_analysis.png")
    return ss


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    data_pairs = build_data_pairs(IMAGE_DIR, MASK_DIR)
    model      = build_model(device)
    model_path = os.path.join(OUTPUT_DIR, "unet_best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    df = extract_biomarkers(data_pairs, model)
    df = standardize(df)
    plot_standardization(df, OUTPUT_DIR)
    df = apply_heckmatt(df, OUTPUT_DIR)
    side_analysis(df, OUTPUT_DIR)
    bland_altman(df, OUTPUT_DIR)
    ss = longitudinal_analysis(df, OUTPUT_DIR)

    df.to_csv(os.path.join(OUTPUT_DIR, "echointensity_results.csv"), index=False)
    ss.to_csv(os.path.join(OUTPUT_DIR, "subject_stats.csv"),         index=False)
    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
