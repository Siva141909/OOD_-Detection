# JEPA-Med-OOD: Self-Supervised Learning for Out-of-Distribution Detection in Medical Imaging

> A Comparative Study of JEPA, MAE, and Supervised Learning for Chest X-Ray OOD Detection

---

## Table of Contents

- [What We Did](#what-we-did)
- [Why We Did It](#why-we-did-it)
- [How We Did It](#how-we-did-it)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Citation](#citation)

---

## What We Did

We built and evaluated a comprehensive framework for **Out-of-Distribution (OOD) detection in medical imaging**, specifically targeting chest X-ray analysis. The project compares three distinct representation learning paradigms:

1. **JEPA (Joint Embedding Predictive Architecture)** — predicts latent representations of masked image regions from context patches, learning high-level semantic features without reconstructing pixels.
2. **MAE (Masked Autoencoder)** — reconstructs pixel-level content from heavily masked inputs, learning fine-grained visual patterns.
3. **Supervised Learning** — standard end-to-end classification training using labeled data.

Each method is evaluated not just for classification accuracy, but critically for its ability to **detect when a model is seeing data it wasn't trained on** — a key safety requirement for clinical AI deployment.

The project also lays the groundwork for extending this approach to **echocardiogram left ventricular segmentation** (EchoNet-Dynamic / EchoNet-Pediatric) as a future task.

---

## Why We Did It

Medical AI models inevitably encounter data from different hospitals, imaging equipment, and patient populations than what they were trained on. When this happens, standard supervised models often produce **overconfident yet incorrect predictions** — a dangerous failure mode in clinical settings.

Key motivations:

- **Distribution shift is unavoidable in clinical deployment.** Variations in imaging equipment, acquisition protocols, patient demographics, and regional disease prevalence all cause subtle but significant shifts.
- **Supervised models are poorly calibrated on OOD data.** They optimize for discriminative class boundaries, not for recognizing when they're operating outside their training distribution.
- **Self-supervised methods learn distributional structure.** By solving pretext tasks without labels, they may capture broader features that better signal when a sample is anomalous.
- **No systematic comparison existed.** Prior medical imaging OOD work focused almost entirely on supervised representations, leaving the potential of JEPA and MAE largely unexplored.

The core research question: *Do self-supervised pretraining paradigms produce representations that are more reliable for OOD detection in medical imaging compared to supervised learning?*

---

## How We Did It

### Datasets

| Role | Dataset | Size | Description |
|------|---------|------|-------------|
| In-Distribution (Train) | TB Chest Radiography Database | 4,200 images (700 normal, 3,500 TB) | Binary labels: Normal / Tuberculosis |
| OOD Test Set 1 | Montgomery County (USA) | 138 images | Acquired on Eureka stationary X-ray machine |
| OOD Test Set 2 | Shenzhen Hospital (China) | 662 images | Acquired on Philips DR Digital Diagnost system |

The OOD sets differ in **geographic origin, patient demographics, imaging equipment, and acquisition protocols** — representing realistic clinical deployment scenarios.

### Architecture

All methods use **Vision Transformer Small (ViT-S)**:
- 12 transformer blocks, embedding dimension 384, 6 attention heads
- ~21.6 million parameters
- Input: 224×224 images divided into 16×16 patches (196 tokens)

### Pretraining Methods

**JEPA:**
- Context encoder processes visible patches; target encoder (EMA copy) processes target patches
- Predictor (6 transformer blocks) predicts target representations from context
- Aggressive masking: 85–100% context masking, 4 target blocks covering 15–20% each
- Loss: L2 distance between predicted and actual target representations (stop-gradient on targets)

**MAE:**
- 75% random patch masking during training
- Encoder processes only visible patches; lightweight decoder (4 blocks, dim 256) reconstructs masked pixels
- Loss: MSE on normalized pixel values in masked regions

**Supervised:**
- End-to-end ViT + linear classification head trained with cross-entropy
- Standard baseline; directly optimizes for classification

### Training Setup

- 50 pretraining epochs, batch size 32, AdamW optimizer (lr=5e-4, cosine annealing)
- Linear probe: 100 epochs, lr=1e-3, frozen encoder
- Data augmentation: random resized crop (scale 0.3–1.0), no horizontal flip (preserves anatomy)
- Single GPU, mixed-precision training

### OOD Scoring Methods

| Method | Description |
|--------|-------------|
| **Energy Score** | Derived from classifier logits; lower energy = more likely in-distribution |
| **Mahalanobis Distance** | Fits class-conditional Gaussians to embedding space; distance to nearest class centroid |

### Robustness Evaluation

Corruptions applied at 6 severity levels (0–5):
- **Gaussian Noise** — simulates sensor/low-dose imaging noise
- **Gaussian Blur** — simulates defocus or motion artifacts
- **Contrast Perturbation** — simulates exposure/window-level variations across institutions

---

## Results

### In-Distribution Classification

| Method | AUROC | Accuracy | ECE | NLL |
|--------|-------|----------|-----|-----|
| JEPA | 0.939 | 91.7% | 0.029 | 0.206 |
| MAE | 0.993 | 97.1% | 0.013 | 0.085 |
| Supervised | **0.996** | **98.0%** | **0.007** | **0.064** |

Supervised learning achieves the best in-distribution performance, as expected from direct optimization for classification. MAE closely follows. JEPA shows slightly lower accuracy, suggesting its focus on semantic predictions may miss some fine-grained TB-specific patterns.

---

### OOD Detection — Energy Score

| Method | Montgomery AUROC | Montgomery FPR@95 | Shenzhen AUROC | Shenzhen FPR@95 |
|--------|-----------------|-------------------|----------------|-----------------|
| JEPA | 0.771 | 0.540 | 0.718 | 0.633 |
| MAE | 0.769 | 0.577 | **0.866** | **0.358** |
| Supervised | 0.607 | 0.814 | 0.582 | 0.931 |

**Key finding:** Self-supervised methods outperform supervised by **16–28 percentage points** in AUROC. The supervised model's FPR@95 of 93.1% on Shenzhen means that when requiring 95% true positive rate, it incorrectly passes nearly all OOD samples — a critical failure for clinical safety.

---

### OOD Detection — Mahalanobis Distance

| Method | Montgomery AUROC | Montgomery FPR@95 | Shenzhen AUROC | Shenzhen FPR@95 |
|--------|-----------------|-------------------|----------------|-----------------|
| JEPA | 0.956 | 0.186 | 0.985 | 0.077 |
| MAE | 0.994 | 0.030 | 0.984 | 0.085 |
| Supervised | 0.991 | 0.024 | **0.997** | **0.000** |

**Key finding:** All methods achieve excellent OOD detection (>95% AUROC) when using Mahalanobis distance. This shows that explicit distribution modeling in embedding space compensates for the weaknesses of confidence-based scoring — even for supervised models.

---

### Cross-Dataset Classification Generalization

| Method | In-Distribution (AUC) | Montgomery (AUC) | Shenzhen (AUC) |
|--------|----------------------|-----------------|----------------|
| JEPA | 0.992 | 0.554 | 0.430 |
| MAE | 0.995 | **0.607** | **0.695** |
| Supervised | **0.996** | 0.471 | 0.592 |

**Key finding:** MAE generalizes significantly better to OOD datasets (10–26 percentage point advantage). JEPA unexpectedly performs near random chance (43–55% AUC), suggesting its latent prediction objective may encode distribution-specific patterns that break down under domain shift.

---

### Robustness to Image Corruptions (AUC at Severity 5)

| Method | Clean | Gaussian Noise | Blur | Contrast |
|--------|-------|---------------|------|----------|
| JEPA | 0.990 | 0.983 | 0.981 | 0.672 |
| MAE | 0.992 | 0.984 | **0.992** | **0.983** |
| Supervised | **0.996** | **0.992** | 0.994 | 0.758 |

**Key finding:** MAE is uniquely robust to contrast perturbations, maintaining **98.3% AUC** under severe contrast shift where JEPA collapses to 67.2% and supervised degrades to 75.8%. This stems from MAE's reconstruction objective requiring the model to predict pixel values across all brightness levels — implicitly learning contrast-invariant features.

---

### Summary of Recommendations

| Scenario | Recommended Approach |
|----------|----------------------|
| Confidence-based OOD detection | MAE or JEPA pretraining + Energy score |
| Highest OOD detection reliability | Any method + Mahalanobis distance |
| Robust to imaging protocol variation (contrast) | MAE pretraining |
| Best in-distribution accuracy | Supervised learning |
| Cross-institution generalization | MAE pretraining |

**Bottom line:** MAE + Mahalanobis distance is the recommended baseline for medical imaging deployment where distribution shift is expected.

---

## Project Structure

```
OOD_Detection/
├── cxr_ood/                        # Chest X-Ray OOD Detection (main experiments)
│   ├── pretraining/
│   │   ├── jepa_pretrain.py        # JEPA pretraining with EMA target encoder
│   │   ├── mae_pretrain.py         # MAE reconstruction pretraining
│   │   └── supervised_pretrain.py  # Supervised classification baseline
│   ├── evaluation/
│   │   ├── linear_probe.py         # Frozen encoder + linear classifier
│   │   ├── ood_detection.py        # Energy & Mahalanobis OOD scoring
│   │   └── embedding_analysis.py   # t-SNE visualization & statistics
│   ├── analysis/
│   │   ├── baseline_comparison.py          # JEPA vs MAE vs Supervised comparison
│   │   ├── robustness_ablation.py          # Corruption robustness testing
│   │   ├── calibration_analysis.py         # ECE & reliability diagrams
│   │   └── embedding_3d_visualization.py   # 3D point cloud visualization
│   ├── utils/
│   │   ├── common.py               # Shared paths & utilities
│   │   ├── cxr_dataset.py          # CXR dataset loader
│   │   └── sanity_check.py         # Checkpoint verification
│   └── configs/
│       └── cxr_vit_small.yaml      # ViT-Small config for CXR experiments
│
├── echo_seg/                       # EchoNet LV Segmentation (planned extension)
│   ├── pretraining/                # JEPA pretraining on echo frames
│   ├── segmentation/               # LV segmentation models
│   ├── utils/                      # Dataset loaders for EchoNet
│   └── configs/
│
├── IJEPA_Meta/                     # Meta's official I-JEPA implementation (base)
│   ├── src/
│   │   ├── train.py                # I-JEPA training loop
│   │   ├── models/                 # ViT architecture definitions
│   │   ├── masks/                  # Masking strategies
│   │   └── utils/                  # Shared utilities
│   ├── main.py                     # Single/multi-GPU local training entry
│   └── main_distributed.py         # SLURM distributed training entry
│
├── paper_final.tex                 # Research paper (LaTeX source)
├── paper_paragraph.tex             # Paper paragraphs draft
├── create_presentation_final.py    # Presentation generation script
└── README.md                       # This file
```

---

## Getting Started

### Requirements

Install dependencies from the included `requirements.txt` (Python 3.8+):

```bash
pip install -r requirements.txt
```

```
torch>=2.0
torchvision
pyyaml
numpy
opencv-python
scikit-learn
matplotlib
pandas
submitit
```

> **Hardware note:** All pretraining scripts default to CUDA when available and fall back to CPU otherwise. There is no dedicated NVIDIA GPU requirement enforced by the code, but ViT-Small pretraining (50 epochs) is impractical on CPU-only or integrated-graphics machines (e.g. Intel Iris Xe) — expect it to be 20-50x slower than on a CUDA GPU. See the "Running on a laptop / no discrete GPU" note below.

### Running Experiments

All scripts must be run from the project root (`OOD_Detection/`), and expect a `Datasets/CXR/` folder with the exact structure described in [Required dataset folder structure](#required-dataset-folder-structure) below.

```bash
# 1. JEPA Pretraining
python cxr_ood/pretraining/jepa_pretrain.py \
  --config cxr_ood/configs/cxr_vit_small.yaml

# 2. MAE Pretraining
python cxr_ood/pretraining/mae_pretrain.py \
  --data-root Datasets/CXR/TB_Chest_Radiography_Database \
  --output-dir experiments/cxr_jepa_pilot/baseline_comparison/mae_pretrain

# 3. Supervised Pretraining
python cxr_ood/pretraining/supervised_pretrain.py \
  --data-root Datasets/CXR/TB_Chest_Radiography_Database \
  --output-dir experiments/cxr_jepa_pilot/baseline_comparison/supervised_pretrain

# 4. Linear Probe Evaluation (requires --data-path, not just --checkpoint)
python cxr_ood/evaluation/linear_probe.py \
  --checkpoint experiments/cxr_jepa_pilot/checkpoint_ep50.pth \
  --data-path Datasets/CXR/TB_Chest_Radiography_Database \
  --output-dir experiments/cxr_jepa_pilot/linear_probe

# 5. OOD Detection (requires jepa+probe checkpoints, data-root, split-file, output-dir)
python cxr_ood/evaluation/ood_detection.py \
  --jepa-checkpoint experiments/cxr_jepa_pilot/checkpoint_ep50.pth \
  --probe-checkpoint experiments/cxr_jepa_pilot/linear_probe/best_probe.pth \
  --data-root Datasets/CXR \
  --split-file experiments/cxr_jepa_pilot/linear_probe/dataset_split.json \
  --output-dir experiments/cxr_jepa_pilot/ood_detection

# 6. Baseline Comparison (jepa-checkpoint + data-root + output-dir are required)
python cxr_ood/analysis/baseline_comparison.py \
  --jepa-checkpoint experiments/cxr_jepa_pilot/checkpoint_ep50.pth \
  --mae-checkpoint experiments/cxr_jepa_pilot/baseline_comparison/mae_pretrain/encoder_final.pth \
  --supervised-checkpoint experiments/cxr_jepa_pilot/baseline_comparison/supervised_pretrain/best_model.pth \
  --data-root Datasets/CXR \
  --output-dir experiments/cxr_jepa_pilot/baseline_comparison

# 7. Robustness Testing (all three checkpoints + data-root + output-dir are required)
python cxr_ood/analysis/robustness_ablation.py \
  --jepa-checkpoint experiments/cxr_jepa_pilot/checkpoint_ep50.pth \
  --mae-checkpoint experiments/cxr_jepa_pilot/baseline_comparison/mae_pretrain/encoder_final.pth \
  --supervised-checkpoint experiments/cxr_jepa_pilot/baseline_comparison/supervised_pretrain/best_model.pth \
  --data-root Datasets/CXR \
  --output-dir experiments/cxr_jepa_pilot/robustness

# 8. 3D Embedding Visualization
python cxr_ood/analysis/embedding_3d_visualization.py \
  --jepa-checkpoint experiments/cxr_jepa_pilot/checkpoint_ep50.pth \
  --mae-checkpoint experiments/cxr_jepa_pilot/baseline_comparison/mae_pretrain/encoder_final.pth \
  --supervised-checkpoint experiments/cxr_jepa_pilot/baseline_comparison/supervised_pretrain/best_model.pth \
  --data-root Datasets/CXR \
  --output-dir experiments/cxr_jepa_pilot/3d_visualization
```

Notes on the fixes above vs. the previous version of this README:
- `cxr_ood/configs/cxr_vit_small.yaml` previously hardcoded a Windows path (`c:/Users/nikhi/Downloads/...`) for `data.root_path` and `logging.folder`. It now uses relative paths (`Datasets/CXR`, `experiments/cxr_jepa_pilot/`), resolved relative to wherever you run the command from — run it from the project root.
- `linear_probe.py` requires `--data-path` (it does not read it from the checkpoint); the old command omitted it and would fail with `error: the following arguments are required: --data-path`.
- `ood_detection.py` takes `--jepa-checkpoint`, `--probe-checkpoint`, `--data-root`, `--split-file`, `--output-dir` — the old command's `--checkpoint` flag does not exist on this script.
- `baseline_comparison.py` and `robustness_ablation.py` both require checkpoint/data-root/output-dir arguments; the old commands passed none and would fail immediately with `argparse` "required" errors.

### Required dataset folder structure

The `Datasets/` folder is intentionally excluded from git (see `.gitignore`) since these medical imaging datasets are large. You need to create it yourself and download the data into it — the code expects this **exact** structure (folder names are hardcoded in the dataset loader classes, e.g. `cxr_ood/utils/cxr_dataset.py`, `cxr_ood/evaluation/ood_detection.py`, `cxr_ood/analysis/baseline_comparison.py`):

```
OOD_Detection/                          <- project root, run all commands from here
└── Datasets/
    └── CXR/
        ├── TB_Chest_Radiography_Database/
        │   ├── Normal/                 <- *.png/.jpg images, label = 0
        │   └── Tuberculosis/           <- *.png/.jpg images, label = 1
        ├── Montgomery TB CXR/          <- exact name, including the space
        │   ├── images/                 <- *.png images
        │   └── montgomery_metadata.csv <- only needed by embedding_analysis.py (columns: study_id, findings)
        └── Shenzhen TB CXR/            <- exact name, including the space
            ├── images/
            │   └── images/             <- yes, nested twice - matches the Kaggle zip layout
            └── shenzhen_metadata.csv   <- only needed by embedding_analysis.py (columns: study_id, findings)
```

Notes:
- `Montgomery TB CXR` and `Shenzhen TB CXR` must match exactly (including the space and capitalization) — a folder named `Montgomery/` or `Shenzhen/` will not be found and those scripts will silently report 0 samples instead of erroring.
- The `montgomery_metadata.csv` / `shenzhen_metadata.csv` files are only required by `embedding_analysis.py` (for per-sample Normal/TB labels used in cross-dataset classification); every other script (`ood_detection.py`, `baseline_comparison.py`, `robustness_ablation.py`, `embedding_3d_visualization.py`) only needs the image folders since it treats Montgomery/Shenzhen as unlabeled OOD data.
- JEPA pretraining (`cxr_ood/utils/cxr_dataset.py`) can optionally also use a `Kermany's CXR/` folder (with `train/`, `test/` splits and `NORMAL/`, `PNEUMONIA/` subfolders) if you add `kermany` to `datasets_to_use` in the config — the default `cxr_vit_small.yaml` only uses `tb_database`, so this is not required for the documented commands above.

### Running on a laptop / no discrete GPU

All training scripts (`jepa_pretrain.py`, `mae_pretrain.py`, `supervised_pretrain.py`) call `torch.cuda.is_available()` and automatically fall back to CPU — nothing crashes on a machine without an NVIDIA GPU. However:

- **Intel Iris Xe (integrated graphics) is not usable by PyTorch here.** PyTorch's CUDA backend only targets NVIDIA GPUs; Iris Xe has no CUDA driver, so training silently runs on CPU regardless of the "graphics card" line in Task Manager.
- **A 13th Gen i5-1335U + 16 GB RAM machine (e.g. `LAPTOP-V5OBE5FT`, Windows 11) can run this code correctly, but CPU-only ViT-Small pretraining is slow.** As a rough estimate, 50 epochs over ~4,200 images that takes ~15-30 minutes on a mid-range CUDA GPU would likely take several hours to a day+ on this CPU alone (highly dependent on batch size, worker count, and thermal throttling on a thin laptop chassis).
- RAM is not the bottleneck (16 GB is comfortable for batch sizes 32-64 at 224×224 with ViT-Small, ~22M params); compute is.
- Practical options on this hardware:
  1. Reduce `epochs`, `batch_size` (e.g. `--batch-size 8-16`), and use `--max_images` (JEPA) / a data subset to sanity-check the pipeline end-to-end quickly.
  2. Run full-scale training on a cloud GPU instead (Google Colab free/paid tier, Kaggle Notebooks with a free T4 GPU, Lightning AI, or a paid instance on AWS/GCP/Lambda/RunPod) and only use the laptop for development, debugging, and running the lightweight evaluation/analysis scripts against already-trained checkpoints.
  3. Evaluation-only scripts (`linear_probe.py` with `--load-model`, `ood_detection.py`, `baseline_comparison.py`, `robustness_ablation.py`, `embedding_3d_visualization.py`) are much cheaper — a frozen encoder forward pass over a few thousand images — and are reasonable to run on this CPU once checkpoints exist.

### Datasets

Download and place under `Datasets/`:
- [TB Chest Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) — in-distribution training data
- [Montgomery County CXR Set](https://openi.nlm.nih.gov/) — OOD test set 1
- [Shenzhen Hospital CXR Set](https://openi.nlm.nih.gov/) — OOD test set 2

---

## Key Takeaways

1. **Self-supervised pretraining is critical for energy-based OOD detection.** JEPA and MAE outperform supervised learning by 15–28 percentage points in AUROC using energy scores.
2. **Mahalanobis distance is a reliable OOD detection method regardless of pretraining.** All methods exceed 95% AUROC, making detection methodology as important as representation choice.
3. **MAE is the most robust method for clinical deployment.** It maintains 98.3% AUC under severe contrast perturbations (vs 67–76% for others) and generalizes best across hospital domains.
4. **JEPA shows a cross-domain generalization limitation.** Despite strong in-distribution performance, it approaches random chance on cross-dataset classification, suggesting it encodes distribution-specific features.
5. **Supervised models require Mahalanobis scoring.** Their confidence estimates are unreliable for OOD data, but their embedding structure still supports excellent Mahalanobis-based detection.

---
