# Computer Vision Intern Assessment: Egg Segmentation & Multi-Object Tracking

An end-to-end computer vision pipeline developed for the CV Internship Assessment, featuring foundation model segmentation (Meta SAM ViT-B), a leakage-aware dataset partition, classical image processing benchmarking (4 progressive trials), quantitative pixel-level evaluation (IoU, Dice, Precision, Recall), multi-object kinematic video tracking, and an automated publication-grade PDF report.

---

## Executive Summary & Key Results

- **Task 1 (Pretrained Foundation Model Segmentation)**: Processed all 110 dataset images using Meta AI's **Segment Anything Model (SAM ViT-B)** accelerated on GPU (682.97s total, ~6.21s/image). Overlapping proposals were filtered via physical geometric and chromatic priors (solidity $\ge 0.84$, circularity $\ge 0.50$, NMS IoU 0.35) to generate 110 binary **pseudo-ground-truth** masks.
- **Leakage-Aware Split**: Discovered that images originate from continuous video frames across 5 capture scenes. Implemented a sequence-block holdout into **88 Development (80%)** and **22 Evaluation (20%)** images, preventing temporal frame leakage.
- **Task 2 (Classical Segmentation Benchmarking)**: Developed and tested 4 progressive classical trials strictly on the Development Split. **Trial 4 (Hybrid Chroma-Gated Otsu + Contour Hole-Filling)** emerged as the winning method with **IoU 0.3903**, **Dice 0.5175**, **Precision 0.4654**, and **Recall 0.8501** at **8.36 ms/frame**.
- **Task 3 (Evaluation & Diagnostics)**: Compared classical segmentation against SAM pseudo-GT. Evaluated Development (IoU 0.3903) vs. held-out Evaluation split (IoU 0.3586). Produced per-scene breakdowns and 5-column qualitative diagnostic error maps (TP Green, FP Red, FN Blue).
- **Task 4 (Multi-Object Video Tracking)**: Retrieved and processed the spinning egg physics demonstration video (640x360 @ 24 fps, 1,440 frames, 60.0s). Deployed a SORT-style multi-object tracker with kinematic velocity coasting, successfully tracking the raw and hard-boiled eggs with persistent IDs across 689 dual-egg frames.
- **Task 5 (SAM Mask-Decoder Fine-Tuning & Continuation)**: Fine-tuned the SAM ViT-B mask decoder (4.06M trainable params; image and prompt encoders frozen) strictly on the 88 development images. Pilot fine-tuning (5 epochs) elevated agreement with SAM pseudo-labels from **IoU 0.7015 to 0.8463** (+0.1448). Controlled continuation (Epochs 6–15) reached **IoU 0.8534** (+0.0071), with 6 improved, 12 neutral, and 4 regressed images. **Epoch 5 (`models/checkpoints/finetuned/checkpoint_epoch_005.pt`) is documented as the recommended checkpoint** due to higher precision (0.9069 vs 0.8993) and multi-egg separation preservation.
- **PDF Report**: Compiled `CV_Intern_Assessment_Report.pdf` with methodology, tables, figures, tracking HUD, fine-tuning diagnostics, and reproducibility guide.

> **Important Methodology Notice**: In accordance with scientific integrity guidelines, all SAM-generated annotations are explicitly designated as **SAM pseudo-ground-truth**, not human-verified ground truth. All fine-tuning and classical metrics quantify agreement against these pseudo-labels.

---

## Quantitative Evaluation Benchmark

### 1. Classical Segmentation Trials (Development Split, N = 88)
| Trial | Core Technique & Parameters | Dev IoU | Dev Dice (F1) | Precision | Recall | Latency | Status & Empirical Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Trial 1** | **Grayscale Global Otsu** (Gaussian $5\times5$ + Closing $7\times7$) | 0.3839 | 0.5137 | 0.4561 | 0.8370 | **1.62 ms** | Discarded: Bleeds into marble veins & water reflections. |
| **Trial 2** | **HSV + CIE-LAB Color-Space** ($L > 125, \text{Chroma} < 35, S < 90, V > 120$) | 0.3518 | 0.4824 | 0.3971 | **0.8663** | 7.83 ms | Discarded: Strict lightness threshold clipped shadowed shells. |
| **Trial 3** | **CLAHE + Adaptive Gaussian + Geometry** ($31\times31, C=-5$, Solidity $\ge 0.82$) | 0.0016 | 0.0029 | 0.0129 | 0.0016 | 6.83 ms | Discarded (Informative Failure): Adaptive Gaussian creates hollow outline rings rejected by area filters. |
| **Trial 4** | **Hybrid Chroma-Gated Otsu + Hole-Filling** (Chroma $< 45, S < 100$, Closing $11\times11$) | **0.3903** | **0.5175** | **0.4654** | 0.8501 | 8.36 ms | **Retained as Final**: Combines Otsu recall with chroma skin/fabric rejection and glare filling. |

### 2. Task 3 Split & Scene Evaluation vs. SAM Pseudo-GT
| Split / Scene Group | Image Count | Mean IoU | Median IoU | Std IoU | Mean Dice | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Development Split (80%)** | 88 | **0.3903** | 0.4431 | 0.2291 | **0.5175** | **0.4654** | **0.8501** |
| **Evaluation Split (20%)** | 22 | **0.3586** | 0.3924 | 0.2312 | **0.4746** | **0.4140** | **0.8211** |
| **Overall Dataset** | 110 | **0.3840** | 0.4357 | 0.2290 | **0.5090** | **0.4551** | **0.8443** |
| `kitchen_countertop_multiple` | 24 | 0.4273 | 0.4764 | 0.2014 | 0.5666 | 0.5061 | 0.8363 |
| `hand_holding_single_egg` | 39 | 0.4159 | 0.4770 | 0.2185 | 0.5279 | 0.4995 | 0.8352 |
| `patterned_cloth_fabric` | 15 | 0.3787 | 0.4518 | 0.2297 | 0.5246 | 0.4857 | 0.7806 |
| `water_metallic_bowl` | 25 | 0.3265 | 0.3241 | 0.2471 | 0.4487 | 0.3647 | 0.8814 |
| `ash_charcoal_bowl` | 7 | 0.2740 | 0.3129 | 0.1764 | 0.3872 | 0.2904 | 0.9262 |

### 3. SAM ViT-B Fine-Tuning & Continuation Benchmark (Held-Out Evaluation Split, N = 22)
| Model / Checkpoint | Mean IoU | Median IoU | Std IoU | Mean Dice | Precision | Recall | Status & Empirical Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Base Pretrained SAM** | 0.7015 | 0.9136 | 0.3490 | 0.7605 | 0.7420 | 0.8120 | Baseline Foundation Weights |
| **Pilot (Epoch 5)** | **0.8463** | 0.9375 | **0.1860** | **0.9040** | **0.9069** | 0.9239 | **RECOMMENDED CHECKPOINT** (Optimal precision & cluster separation) |
| **Continued (Epoch 15)** | 0.8534 | **0.9432** | 0.1997 | 0.9053 | 0.8993 | **0.9386** | Preserved; exhibits multi-egg boundary spill and reflection artifacts |
| **Delta (Ep 15 - Ep 5)** | **+0.0071** | +0.0057 | +0.0137 | +0.0013 | -0.0076 | +0.0147 | 6 Improved \| 12 Neutral \| 4 Regressed |


---

## Project Structure

```
internship-assignment/
├── configs/
│   └── config.yaml                     # Primary hyperparameter and path configuration
├── data/
│   ├── splits.json                     # Leakage-aware 80:20 sequence-block partition
│   ├── task1_sam_masks/                # SAM ViT-B binary masks & visualizations
│   │   ├── masks/                      # 110 binary PNG masks (640x640 uint8)
│   │   ├── visualizations/             # 110 emerald overlay verification PNGs
│   │   └── metadata.json               # Manifest recording flags, status, areas
│   └── task2_classical_masks/          # Classical segmentation outputs
│       ├── final/                      # 110 final prediction masks (Trial 4)
│       └── experiment_log.json         # Structured log of 4 development trials
├── models/
│   └── checkpoints/
│       └── sam_vit_b_01ec64.pth        # Official Meta SAM ViT-B weights (357.7 MB)
├── outputs/
│   ├── figures/                        # Diagnostic and analytical figures
│   │   ├── metrics_distribution.png    # Scene bar plot & Dev/Eval boxplot
│   │   ├── qualitative_analysis.png    # 5-column success vs failure diagnostic panel
│   │   ├── qualitative_successes.png   # Success diagnostic panels
│   │   ├── qualitative_failures.png    # Failure diagnostic panels
│   │   └── tracking_sample_frame_*.png # Rendered video tracking frame samples
│   ├── evaluation_analysis.md          # Comprehensive Task 3 technical report
│   ├── evaluation_metrics.json         # Per-image and aggregate metrics JSON
│   ├── tracking_summary.json           # Video tracking performance telemetry
│   └── tracked_video.mp4               # Rendered multi-object tracking video (7.66 MB)
├── src/
│   ├── egg_filter.py                   # Geometric & photometric egg candidate filter
│   ├── task1_pretrained.py             # Pretrained SAM segmentation pipeline
│   ├── task2_classical.py              # Classical image-processing trials (1-4)
│   ├── task3_evaluation.py             # Quantitative & qualitative evaluation
│   ├── task4_tracking.py               # Multi-object video tracker (SORT + coasting)
│   ├── generate_pdf_report.py          # ReportLab PDF report compiler
│   ├── create_splits.py                # Sequence-block split generator
│   └── smoke_test.py                   # Minimal end-to-end pipeline verification
├── run_pipeline.py                     # Master pipeline orchestrator (Tasks 1-5)
├── requirements.txt                    # Pinned Python package dependencies
├── CV_Intern_Assessment_Report.pdf     # Formal PDF assessment submission report
└── README.md                           # Documentation and reproduction guide
```

---

## Environment Setup & Reproduction

### 1. Hardware & Software Requirements
- **OS**: Windows 11 (tested) / Linux (Ubuntu 22.04+)
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (or any CUDA-compatible GPU with $\ge 4$ GB VRAM)
- **Python**: Python 3.12.7
- **PyTorch**: PyTorch 2.5.1 + CUDA 12.1

### 2. Virtual Environment Setup
```powershell
# Create virtual environment leveraging existing CUDA PyTorch packages
uv venv --system-site-packages .venv
.venv\Scripts\activate

# Install project dependencies
uv pip install -r requirements.txt
```

### 3. Execution Commands

#### Option A: Run Entire End-to-End Pipeline
```powershell
.venv\Scripts\python.exe run_pipeline.py
```
*Note: `run_pipeline.py` respects incremental caches, skipping pre-computed SAM masks and heavy operations.*

#### Option B: Run Individual Modules Step-by-Step
```powershell
# 1. Foundation Pretrained Segmentation (SAM ViT-B)
.venv\Scripts\python.exe -m src.task1_pretrained --config configs/config.yaml

# 2. Classical Segmentation Trials (Dev Split) & Final Mask Export
.venv\Scripts\python.exe src/task2_classical.py

# 3. Quantitative & Qualitative Evaluation vs SAM Pseudo-GT
.venv\Scripts\python.exe src/task3_evaluation.py

# 4. Multi-Object Video Tracking (YouTube Physics Demo)
.venv\Scripts\python.exe src/task4_tracking.py

# 5. Evaluate Fine-Tuned SAM Checkpoints (Base vs Ep 5 vs Ep 15)
.venv\Scripts\python.exe -m src.eval_continuation

# 6. Generate Publication-Grade PDF Report
.venv\Scripts\python.exe src/generate_pdf_report.py
```

### Documented Model Checkpoint Paths
- **Recommended Checkpoint**: `models/checkpoints/finetuned/checkpoint_epoch_005.pt` (Epoch 5, 47.8 MB) — Retained as recommended due to optimal precision (0.9069) and multi-egg separation.
- **Continuation Checkpoint**: `models/checkpoints/finetuned/continuation_10_epochs/checkpoint_final.pt` (Epoch 15, 47.8 MB) — Preserved for ablation comparison.
- **Base Foundation SAM**: `models/checkpoints/sam_vit_b_01ec64.pth` (Official Meta ViT-B, 357.7 MB) — Frozen baseline weights.

---

## Task 4 Video Tracking Demonstration

- **Demonstration Video**: [YouTube: Raw vs Cooked Egg Rotational Inertia Demonstration](https://www.youtube.com/watch?v=Y4JyV94y2_A)
- **Properties**: 640×360 resolution, 24.0 FPS, 1,440 frames, 60.00s duration.
- **Tracking Approach**: Classical color-space detector combined with SORT-style Euclidean distance centroid matching and linear kinematic velocity coasting during momentary hand occlusions.
- **Tracked Artifact**: Saved locally to `outputs/tracked_video.mp4` (7.66 MB).
- **Public Video Link**: Pending configuration (public cloud storage link to be attached upon deployment; locally available at `outputs/tracked_video.mp4`).

---

## AI Tools Disclosure

In compliance with internship assessment guidelines, AI coding assistants (Claude 3.7 Sonnet / Google DeepMind Antigravity) were utilized for rapid code scaffolding, ReportLab styling, and CLI boilerplate. All architectural decisions, mathematical metric implementations, color-space gating parameters, morphological kernels, tracking Kalman/coasting filters, and empirical analysis were engineered and validated directly by the candidate.

