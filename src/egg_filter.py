"""
Egg Mask Candidate Filtering and Selection Module.
Filters generic SAM mask proposals using geometric and photometric priors specific to eggs:
- Smoothness and convexity (high solidity)
- Elliptic compactness (circularity and aspect ratio)
- Scale consistency (area bounding)
- Photometric properties (pale / high brightness, low-to-moderate saturation)
- Non-Maximum Suppression (NMS) for overlapping proposals
"""

import cv2
import numpy as np


def compute_mask_properties(mask: np.ndarray, image_bgr: np.ndarray) -> dict:
    """Compute geometric and photometric metrics for a single binary mask."""
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Take the largest external contour
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if area < 10 or perimeter < 10:
        return None
    
    # 1. Convexity / Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    
    # 2. Circularity / Isoperimetric quotient
    circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
    
    # 3. Aspect Ratio via Fitted Ellipse or MinAreaRect
    aspect_ratio = 1.0
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (d1, d2), angle = ellipse
            major_axis = max(d1, d2)
            minor_axis = min(d1, d2)
            aspect_ratio = minor_axis / major_axis if major_axis > 0 else 1.0
        except Exception:
            rect = cv2.minAreaRect(contour)
            w_r, h_r = rect[1]
            aspect_ratio = min(w_r, h_r) / max(w_r, h_r) if max(w_r, h_r) > 0 else 1.0
    else:
        rect = cv2.minAreaRect(contour)
        w_r, h_r = rect[1]
        aspect_ratio = min(w_r, h_r) / max(w_r, h_r) if max(w_r, h_r) > 0 else 1.0
        
    # 4. Photometric properties (HSV & Gray)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    mask_bool = mask > 0
    mean_val = float(np.mean(hsv[mask_bool, 2])) if np.any(mask_bool) else 0.0
    mean_sat = float(np.mean(hsv[mask_bool, 1])) if np.any(mask_bool) else 0.0
    mean_gray = float(np.mean(gray[mask_bool])) if np.any(mask_bool) else 0.0
    
    return {
        "area": area,
        "perimeter": perimeter,
        "solidity": solidity,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "mean_val": mean_val,
        "mean_sat": mean_sat,
        "mean_gray": mean_gray,
        "contour": contour
    }


def filter_egg_candidates(
    sam_masks: list,
    image_bgr: np.ndarray,
    config: dict = None
) -> list:
    """
    Filter and score SAM candidates against egg priors.
    Returns list of candidate dicts sorted by egg likeness score.
    """
    cfg = config or {}
    min_area = cfg.get("min_area", 5000)
    max_area = cfg.get("max_area", 200000)
    min_solidity = cfg.get("min_solidity", 0.85)
    min_circularity = cfg.get("min_circularity", 0.55)
    min_aspect_ratio = cfg.get("min_aspect_ratio", 0.40)
    max_aspect_ratio = cfg.get("max_aspect_ratio", 0.98)
    min_brightness = cfg.get("min_mean_brightness", 90)
    max_saturation = cfg.get("max_mean_saturation", 125)
    
    scored_candidates = []
    
    for candidate in sam_masks:
        mask = candidate["segmentation"]
        props = compute_mask_properties(mask, image_bgr)
        if not props:
            continue
            
        area = props["area"]
        solidity = props["solidity"]
        circ = props["circularity"]
        ar = props["aspect_ratio"]
        b_val = props["mean_val"]
        s_val = props["mean_sat"]
        
        # Hard filtering criteria
        if not (min_area <= area <= max_area):
            continue
        if solidity < min_solidity:
            continue
        if circ < min_circularity:
            continue
        if not (min_aspect_ratio <= ar <= max_aspect_ratio):
            continue
        if b_val < min_brightness or s_val > max_saturation:
            continue
            
        # Composite egg-likeness score
        pred_iou = candidate.get("predicted_iou", 1.0)
        stability = candidate.get("stability_score", 1.0)
        
        # Score emphasizes solidity, circularity, stability, and brightness
        score = (
            0.30 * solidity +
            0.25 * min(circ, 1.0) +
            0.20 * stability +
            0.15 * pred_iou +
            0.10 * (b_val / 255.0)
        )
        
        scored_candidates.append({
            "mask": mask,
            "score": score,
            "props": props,
            "raw_sam": candidate
        })
        
    # Sort candidates descending by score
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # Non-Maximum Suppression (NMS) to avoid overlapping duplicate masks of same egg
    final_candidates = []
    for cand in scored_candidates:
        mask_a = cand["mask"]
        overlap = False
        for kept in final_candidates:
            mask_b = kept["mask"]
            intersection = np.logical_and(mask_a, mask_b).sum()
            union = np.logical_or(mask_a, mask_b).sum()
            iou = intersection / union if union > 0 else 0
            if iou > 0.35:  # Overlapping candidate for the same egg
                overlap = True
                break
        if not overlap:
            final_candidates.append(cand)
            
    return final_candidates
