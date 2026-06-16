# Standardization of Ultrasound-Based Muscle Biomarkers
### Characterizing Echointensity of the Lumbar Multifidus Muscle

**AI for Healthcare | Semester 6 | National Institute of Technology Calicut | May 2026**

> **Author:** Shiva Kumar (B231244EE)  
> **Guided by:** Dr. Jayaraj PB and Ninitha Mary (PhD Scholar)

---

## Overview

This project presents a complete pipeline for automated segmentation and standardization of **Echo Intensity (EI)** biomarkers from B-mode ultrasound images of the **Lumbar Multifidus (LM)** muscle. Raw EI values vary across scanners, operators, and subjects — making direct comparison unreliable. This pipeline addresses that with deep learning segmentation + multi-method standardization.

**Key results on the LUMINOUS database (341 images, 109 subjects):**

| Metric | Score |
|--------|-------|
| Dice Coefficient | **0.912** |
| IoU Score | **0.843** |
| Precision | **0.922** |
| Recall | **0.912** |
| F1 Score | **0.917** |
| Left–Right EI Asymmetry | **15.93%** |
| Within-subject Longitudinal CV | **13.48%** |

---

## Pipeline at a Glance

```
LUMINOUS Dataset (341 B-mode images)
        ↓
Dataset Analysis (mask type detection)
        ↓
Midpoint-Crop Strategy ← KEY INNOVATION
(dual-mask: split at x=410; single-mask: full image)
        ↓
Data Augmentation (training only)
        ↓
U-Net Segmentation (Dice=0.912)
        ↓
Biomarker Extraction (EI, CSA)
        ↓
EI Standardization (4 methods)
        ↓
Heckmatt Grading + Asymmetry + Longitudinal Analysis
        ↓
Standardized Clinical Report
```

---

## Key Innovation: Midpoint-Crop Strategy

45 of the 341 images contain **both** left and right LM muscles simultaneously. Without correction, the U-Net sees two muscles but predicts only one → double-blob artifacts (Dice ~0.60) and identical Left/Right EI (0% asymmetry artifact).

**Fix:**
- `Mask1` (RIGHT LM) → crop image to RIGHT HALF: columns `[410:820]`
- `Mask2` (LEFT LM) → crop image to LEFT HALF: columns `[0:410]`
- Single mask → use FULL IMAGE (no crop)

**Verified:** Minimum muscle gap = 20 px, mean = 50 px. Midpoint x=410 always falls cleanly between both muscles across all 45 dual-mask pairs.

**Result:** Dice improved from 0.800 → 0.912 (+14%). True biological asymmetry of 15.93% revealed (was 0% before fix).

---

## Repository Structure

```
lumbar-ei-standardization/
├── notebooks/
│   └── echointensity.ipynb       # Full 15-step pipeline notebook
├── src/
│   ├── dataset.py                # LuminousDataset class + midpoint-crop logic
│   ├── model.py                  # U-Net architecture (DoubleConv, UNET)
│   ├── loss.py                   # Combined BCE + Dice loss
│   ├── train.py                  # Training loop with early stopping
│   ├── evaluate.py               # Evaluation metrics (Dice, IoU, Precision, Recall)
│   ├── biomarkers.py             # EI + CSA extraction from predicted masks
│   ├── standardize.py            # 4 standardization methods
│   └── analysis.py               # Heckmatt grading, asymmetry, longitudinal analysis
├── docs/
│   └── project_report.pdf        # Full project report
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

This project uses the **LUMINOUS database** (Belasso et al., 2020, *BMC Musculoskeletal Disorders*).

| Property | Value |
|----------|-------|
| Total B-mode images | 341 |
| Image resolution | 820 × 614 px |
| Total subjects | 109 young athletic adults |
| Multi-session subjects | 107 / 109 (up to 4 sessions) |
| Single-mask images | 296 |
| Dual-mask images | 45 (both LMs annotated) |

> **Download:** [LUMINOUS Database — IMPACT Laboratory](https://share.google/CLO5k57YLlWubbj7T)

---

## U-Net Architecture

- **Encoder:** 4 stages — DoubleConv blocks at 64, 128, 256, 512 channels with MaxPool 2×2
- **Bottleneck:** DoubleConv 1024 ch with Dropout(0.2)
- **Decoder:** ConvTranspose2d 1×2 upsampling + skip connections
- **Loss:** `0.5 × BCE + 0.5 × Dice` (handles class imbalance)
- **Input:** 256×256 px, grayscale
- **Parameters:** ~31M

---

## EI Standardization Methods

| Method | Formula | Best Use | CV |
|--------|---------|----------|----|
| Z-score | `(EI − μ) / σ` | Research / ML features | — |
| Min-Max | `(EI − min) / range` | Visualization / NN input | 41.1% |
| Phantom-reference | `(EI / mean) × 100%` | Clinical decisions | 29.7% |
| Robust | `(EI − median) / IQR` | Outlier-resistant | — |

**Clinical recommendation:** Use Phantom-reference for clinical decisions (values >130% suggest hyperechogenicity). Use Z-score for research and statistical comparisons.

---

## Heckmatt Grading Results

| Grade | Count | % | Clinical Meaning |
|-------|-------|---|-----------------|
| Grade I — Normal | 128 | 33.2% | Normal muscle quality |
| Grade II — Mild | 140 | 36.3% | Slight EI increase |
| Grade III — Moderate | 86 | 22.3% | Marked EI increase |
| Grade IV — Severe | 32 | 8.3% | Very high EI |

Grade I + II = **69.5%** — consistent with a young athletic population.

---

## Installation & Usage

### Requirements

```bash
pip install -r requirements.txt
```

### Running the Full Pipeline

Open and run the notebook:

```bash
jupyter notebook notebooks/echointensity.ipynb
```

Or run the modular scripts:

```bash
# Set your dataset path
export LUMINOUS_DIR=~/LUMINOUS_Database

# Train the U-Net
python src/train.py

# Extract biomarkers and run analysis
python src/biomarkers.py
python src/analysis.py
```

### Hardware

Trained on **Tesla V100 (32 GB VRAM)** with mixed precision (FP16). Runs on any CUDA GPU; reduce `BATCH_SIZE` for smaller GPUs.

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (lr=1e-4, weight_decay=1e-5) |
| LR Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Loss | 0.5 × BCE + 0.5 × Dice |
| Batch Size | 8 |
| Max Epochs | 50 (early stopping patience=10) |
| Input Size | 256 × 256 px |
| Best Epoch | 44 |

---

## Results Summary

### Segmentation (vs. baseline without midpoint-crop)

| Metric | With Fix | Improvement |
|--------|----------|-------------|
| Dice | 0.912 | +14.0% |
| IoU | 0.843 | +21.1% |
| Precision | 0.922 | +27.2% |
| Recall | 0.912 | −0.1% |

### Left vs. Right LM Asymmetry

| Measurement | Left LM | Right LM |
|-------------|---------|----------|
| Mean EI (raw) | 0.238 ± 0.077 | 0.259 ± 0.079 |
| Mean CSA | 4.23 cm² | 3.55 cm² |
| EI Asymmetry | **15.93%** (was 0% artifact) | p = 0.1726 |

---

## Output Files

After running the pipeline, results are saved to `~/unet_improved_final/`:

```
unet_best.pth                    # Trained model weights
echointensity_results.csv        # Per-sample EI, CSA, grades
subject_stats.csv                # Per-subject longitudinal stats
side_asymmetry.csv               # Left vs. Right comparison
training_history.csv             # Epoch-by-epoch loss and Dice
sample_images.png                # Dataset samples with crop visualization
training_curves.png              # Loss and Dice training curves
segmentation_scores.png          # Metric distributions
predictions.png                  # Qualitative prediction examples
standardization_comparison.png   # 4-method EI comparison
heckmatt_grading.png             # Grade distribution
side_analysis.png                # Left vs. Right boxplots
bland_altman.png                 # Raw vs. Phantom agreement
longitudinal_analysis.png        # Multi-session EI/CSA trends
```

---

## References

1. 1. Belasso, C.J., et al. (2020). LUMINOUS: Lumbar Ultrasound Muscle ImagiNG with Open-source analysis software. [Dataset](https://share.google/CLO5k57YLlWubbj7T)
2. Heckmatt, J.Z., Leeman, S., & Dubowitz, V. (1982). Ultrasound imaging in the diagnosis of muscle disease. *Journal of Pediatrics*, 101(5), 656–660.
3. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI 2015*, LNCS 9351, 234–241.
4. Bland, J.M., & Altman, D.G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet*, 327(8476), 307–310.

---

## License

This project is for academic purposes (NIT Calicut, AI for Healthcare, 2026). The LUMINOUS dataset is subject to its own terms — please cite Belasso et al. (2020) when using it.
