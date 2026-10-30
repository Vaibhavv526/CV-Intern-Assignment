# Empirical Evaluation: Base SAM ViT-B vs Fine-Tuned SAM Mask Decoder

## 1. Evaluation Methodology and Caveats

> **CRITICAL METHODOLOGICAL CAVEAT**: This dataset contains **no human-annotated segmentation masks**. All metrics below measure agreement against SAM-generated pseudo-labels (`data/task1_sam_masks/masks/`). Higher agreement with pseudo-labels does **not** prove superior real-world generalization; it specifically measures the fine-tuned decoder's fidelity to prompt-conditioned pseudo-target representations.

- **Test Set**: Exactly the 22 held-out evaluation images from `data/splits.json`.
- **Prompting**: Identical bounding-box prompts extracted from pseudo-label boundaries.
- **Preprocessing & Postprocessing**: Bitwise identical `ResizeLongestSide(1024)` and bilinear unpadding.

## 2. Aggregate Quantitative Comparison

| Metric | Base Pretrained SAM | Fine-Tuned SAM (Decoder-Only) | Delta (FT - Base) |
| :--- | :---: | :---: | :---: |
| **Mean IoU** | 0.7015 ± 0.3490 | 0.8463 ± 0.1860 | **+0.1448** |
| **Median IoU** | 0.9136 | 0.9375 | +0.0239 |
| **Mean Dice** | 0.7605 ± 0.3164 | 0.9040 ± 0.1270 | **+0.1435** |
| **Median Dice** | 0.9548 | 0.9677 | +0.0129 |
| **Mean Precision** | 0.8013 | 0.9069 | +0.1056 |
| **Mean Recall** | 0.7462 | 0.9239 | +0.1777 |
| **Inference Latency** | 448.54 ms | 452.69 ms | +4.15 ms |

## 3. Per-Image Breakdown (22 Evaluation Samples)

| Image Stem | Scene | Base IoU | FT IoU | Delta IoU | Base Dice | FT Dice | Delta Dice |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `not_damaged_78_jpg.rf.d3a...` | ash_charcoal_bowl | 0.9934 | 0.9920 | -0.0014 | 0.9967 | 0.9960 | -0.0007 |
| `not_damaged_95_jpg.rf.080...` | ash_charcoal_bowl | 0.9980 | 0.9981 | +0.0001 | 0.9990 | 0.9990 | +0.0000 |
| `not_damaged_37_jpg.rf.73b...` | hand_holding_single_egg | 0.9188 | 0.9649 | +0.0460 | 0.9577 | 0.9821 | +0.0244 |
| `not_damaged_39_jpg.rf.81c...` | hand_holding_single_egg | 0.9976 | 0.9973 | -0.0003 | 0.9988 | 0.9986 | -0.0001 |
| `not_damaged_40_jpg.rf.71d...` | hand_holding_single_egg | 0.9949 | 0.9920 | -0.0029 | 0.9974 | 0.9960 | -0.0015 |
| `not_damaged_41_jpg.rf.630...` | hand_holding_single_egg | 0.6921 | 0.7151 | +0.0229 | 0.8181 | 0.8339 | +0.0158 |
| `not_damaged_42_jpg.rf.662...` | hand_holding_single_egg | 0.2572 | 0.4541 | +0.1968 | 0.4092 | 0.6245 | +0.2154 |
| `not_damaged_43_jpg.rf.66c...` | hand_holding_single_egg | 0.5967 | 0.6569 | +0.0602 | 0.7474 | 0.7929 | +0.0455 |
| `not_damaged_44_jpg.rf.a4d...` | hand_holding_single_egg | 0.9084 | 0.7792 | -0.1292 | 0.9520 | 0.8759 | -0.0761 |
| `not_damaged_45_jpg.rf.afe...` | hand_holding_single_egg | 0.9961 | 0.9961 | +0.0001 | 0.9980 | 0.9981 | +0.0000 |
| `not_damaged_145_jpg.rf.51...` | kitchen_countertop_multiple | 0.0062 | 0.3803 | +0.3741 | 0.0124 | 0.5511 | +0.5387 |
| `not_damaged_146_jpg.rf.a7...` | kitchen_countertop_multiple | 0.9973 | 0.9978 | +0.0005 | 0.9987 | 0.9989 | +0.0003 |
| `not_damaged_147_jpg.rf.f9...` | kitchen_countertop_multiple | 0.0859 | 0.5675 | +0.4816 | 0.1582 | 0.7241 | +0.5659 |
| `not_damaged_148_jpg.rf.8a...` | kitchen_countertop_multiple | 0.9883 | 0.9953 | +0.0070 | 0.9941 | 0.9976 | +0.0035 |
| `not_damaged_149_jpg.rf.c3...` | kitchen_countertop_multiple | 0.7508 | 0.8699 | +0.1191 | 0.8576 | 0.9304 | +0.0728 |
| `not_damaged_163_jpg.rf.ee...` | patterned_cloth_fabric | 0.5545 | 0.6895 | +0.1350 | 0.7134 | 0.8162 | +0.1028 |
| `not_damaged_164_jpg.rf.ae...` | patterned_cloth_fabric | 0.1029 | 0.8272 | +0.7243 | 0.1865 | 0.9054 | +0.7189 |
| `not_damaged_120_jpg.rf.df...` | water_metallic_bowl | 0.9940 | 0.9940 | -0.0000 | 0.9970 | 0.9970 | -0.0000 |
| `not_damaged_121_jpg.rf.05...` | water_metallic_bowl | 0.2150 | 0.9367 | +0.7217 | 0.3540 | 0.9673 | +0.6134 |
| `not_damaged_122_jpg.rf.b4...` | water_metallic_bowl | 0.9958 | 0.9825 | -0.0133 | 0.9979 | 0.9912 | -0.0067 |
| `not_damaged_123_jpg.rf.11...` | water_metallic_bowl | 0.4452 | 0.8939 | +0.4487 | 0.6161 | 0.9440 | +0.3279 |
| `not_damaged_124_jpg.rf.93...` | water_metallic_bowl | 0.9442 | 0.9383 | -0.0059 | 0.9713 | 0.9682 | -0.0031 |

## 4. Qualitative Visualizations

Qualitative comparison saved to: `outputs/figures/finetuned_vs_base_qualitative.png`.
