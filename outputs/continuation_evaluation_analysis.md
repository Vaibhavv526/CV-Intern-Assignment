# Controlled Continuation Analysis: Pilot (Epoch 5) vs Continued (Epoch 15)

## 1. Methodology & Split Integrity

> **METHODOLOGICAL NOTE**: All evaluations were performed on the exact **22 held-out evaluation images** from `data/splits.json`. These images were strictly excluded from both the pilot run and the 10-epoch continuation. All metrics reflect agreement with **SAM-generated pseudo-labels**, not human annotations.

## 2. Quantitative Comparison Across Checkpoints

| Model / Checkpoint | Mean IoU | Median IoU | Mean Dice | Median Dice | Mean Precision | Mean Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Pretrained SAM** | 0.7015 | 0.9136 | 0.7605 | 0.9548 | - | - |
| **Pilot (Epoch 5)** | 0.8463 | 0.9375 | 0.9040 | 0.9677 | 0.9069 | 0.9239 |
| **Continued (Epoch 15)** | 0.8534 | 0.9432 | 0.9053 | 0.9708 | 0.8993 | 0.9386 |
| **Delta (Ep 15 - Ep 5)** | **+0.0071** | - | **+0.0013** | - | - | - |

**Outcome Summary**: 6 images improved ($\Delta > +0.01$), 12 remained neutral ($|\Delta| \le 0.01$), and 4 regressed ($\Delta < -0.01$).

## 3. Per-Image Breakdown & Regression Analysis

| Image Stem | Scene | Base IoU | Pilot Ep5 IoU | Cont Ep15 IoU | Delta (15 - 5) | Regression Flag |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `not_damaged_78_jpg.rf.d3a...` | ash_charcoal_bowl | 0.9934 | 0.9920 | 0.9945 | +0.0025 | — |
| `not_damaged_95_jpg.rf.080...` | ash_charcoal_bowl | 0.9980 | 0.9981 | 0.9983 | +0.0003 | — |
| `not_damaged_37_jpg.rf.73b...` | hand_holding_single_egg | 0.9188 | 0.9649 | 0.9680 | +0.0031 | — |
| `not_damaged_39_jpg.rf.81c...` | hand_holding_single_egg | 0.9976 | 0.9973 | 0.9979 | +0.0007 | — |
| `not_damaged_40_jpg.rf.71d...` | hand_holding_single_egg | 0.9949 | 0.9920 | 0.9842 | -0.0078 | — |
| `not_damaged_41_jpg.rf.630...` | hand_holding_single_egg | 0.6921 | 0.7151 | 0.7047 | -0.0104 | ⚠️ REGRESSED |
| `not_damaged_42_jpg.rf.662...` | hand_holding_single_egg | 0.2572 | 0.4541 | 0.4500 | -0.0041 | — |
| `not_damaged_43_jpg.rf.66c...` | hand_holding_single_egg | 0.5967 | 0.6569 | 0.6990 | +0.0421 | ✅ IMPROVED |
| `not_damaged_44_jpg.rf.a4d...` | hand_holding_single_egg | 0.9084 | 0.7792 | 0.8281 | +0.0490 | ✅ IMPROVED |
| `not_damaged_45_jpg.rf.afe...` | hand_holding_single_egg | 0.9961 | 0.9961 | 0.9961 | -0.0001 | — |
| `not_damaged_145_jpg.rf.51...` | kitchen_countertop_multiple | 0.0062 | 0.3803 | 0.2784 | -0.1019 | ⚠️ REGRESSED |
| `not_damaged_146_jpg.rf.a7...` | kitchen_countertop_multiple | 0.9973 | 0.9978 | 0.9981 | +0.0003 | — |
| `not_damaged_147_jpg.rf.f9...` | kitchen_countertop_multiple | 0.0859 | 0.5675 | 0.5039 | -0.0636 | ⚠️ REGRESSED |
| `not_damaged_148_jpg.rf.8a...` | kitchen_countertop_multiple | 0.9883 | 0.9953 | 0.9959 | +0.0006 | — |
| `not_damaged_149_jpg.rf.c3...` | kitchen_countertop_multiple | 0.7508 | 0.8699 | 0.9456 | +0.0758 | ✅ IMPROVED |
| `not_damaged_163_jpg.rf.ee...` | patterned_cloth_fabric | 0.5545 | 0.6895 | 0.8693 | +0.1798 | ✅ IMPROVED |
| `not_damaged_164_jpg.rf.ae...` | patterned_cloth_fabric | 0.1029 | 0.8272 | 0.8431 | +0.0159 | ✅ IMPROVED |
| `not_damaged_120_jpg.rf.df...` | water_metallic_bowl | 0.9940 | 0.9940 | 0.9940 | -0.0000 | — |
| `not_damaged_121_jpg.rf.05...` | water_metallic_bowl | 0.2150 | 0.9367 | 0.8617 | -0.0750 | ⚠️ REGRESSED |
| `not_damaged_122_jpg.rf.b4...` | water_metallic_bowl | 0.9958 | 0.9825 | 0.9898 | +0.0073 | — |
| `not_damaged_123_jpg.rf.11...` | water_metallic_bowl | 0.4452 | 0.8939 | 0.9408 | +0.0469 | ✅ IMPROVED |
| `not_damaged_124_jpg.rf.93...` | water_metallic_bowl | 0.9442 | 0.9383 | 0.9325 | -0.0059 | — |
