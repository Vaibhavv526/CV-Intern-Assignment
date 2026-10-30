# Task 3: Quantitative & Qualitative Evaluation Analysis

## 1. Metric Definitions & Aggregation Methodology

Segmentation quality is evaluated against Meta SAM ViT-B **pseudo-ground-truth** masks using standard computer vision metrics computed at the pixel level:

- **Intersection over Union (IoU / Jaccard Index)**: $\text{IoU} = \frac{|P \cap G|}{|P \cup G|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$
- **Dice Coefficient (F1 Score)**: $\text{Dice} = \frac{2 |P \cap G|}{|P| + |G|} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$
- **Precision**: $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$ (proportion of predicted egg pixels that are truly egg)
- **Recall**: $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$ (proportion of true egg pixels successfully recovered)

> **Critical Methodology Note**: All reported metrics measure agreement against SAM-generated masks. They represent **pseudo-ground-truth agreement**, not verified human ground truth.

## 2. Quantitative Results: Development vs. Evaluation Splits

| Split | Images (N) | Mean IoU | Median IoU | Std IoU | Mean Dice (F1) | Mean Precision | Mean Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Development Set (80%)** | 88 | 0.3903 | 0.3198 | 0.2489 | 0.5175 | 0.4654 | 0.8501 |
| **Evaluation Set (20%)** | 22 | 0.3586 | 0.2995 | 0.2667 | 0.4746 | 0.4141 | 0.8210 |
| **Overall Dataset** | 110 | 0.3840 | 0.3198 | 0.2529 | 0.5090 | 0.4551 | 0.8443 |

The evaluation set exhibits a slight drop in mean IoU (0.3903 vs. 0.3586), reflecting realistic domain shift across holdout temporal sequences without artificial hyperparameter overfitting.

## 3. Per-Scene Breakdown & Empirical Analysis

| Visual Scene Category | Count | Mean IoU | Mean Dice | Precision | Recall | Primary Scene Challenge |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `ash_charcoal_bowl` | 7 | 0.2740 | 0.3872 | 0.2904 | 0.9262 | High recall (0.9262) but bowl interior reflections lower precision. |
| `hand_holding_single_egg` | 39 | 0.4159 | 0.5279 | 0.4995 | 0.8352 | Strong balance; finger occlusions partially misclassified as egg boundary. |
| `kitchen_countertop_multiple` | 24 | 0.4273 | 0.5666 | 0.5061 | 0.8363 | Highest overall IoU (0.4273); white marble veins create false positives. |
| `patterned_cloth_fabric` | 15 | 0.3787 | 0.5246 | 0.4857 | 0.7806 | Weave texture suppressed by chroma gating; fabric shadows create small edge cuts. |
| `water_metallic_bowl` | 25 | 0.3265 | 0.4487 | 0.3647 | 0.8814 | Water ripples and specular glare cause moderate over-segmentation. |

## 4. Edge Cases & Qualitative Failure Diagnostics

1. **Marble Countertop & Reflection Bleed**: White veins in kitchen marble share the neutral high-value signature of white egg shells, producing false positive clusters.
2. **Finger Occlusions**: In the hand-holding sequence, human fingers wrap around the egg. While chroma gating suppresses orange/red skin tones, fingertips casting shadows onto the shell alter local hue.
3. **Water Refraction & Specular Highlights**: In metallic and glass bowls, specular light reflection creates saturated hot spots. Contour hole-filling successfully bridges these highlights in the winning Trial 4 pipeline.

