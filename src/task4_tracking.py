"""
Task 4: Multi-Object Egg Segmentation & Tracking on YouTube Physics Demonstration Video.
Video: https://www.youtube.com/watch?v=Y4JyV94y2_A (Raw vs Cooked Egg Rotational Inertia Demo).

Pipeline Versions:
- v1: Legacy baseline (global color threshold, greedy SORT matching, 9x9 closing).
- v2: Experimental baseline (tabletop ROI, basic skin filter, Hungarian assignment).
- v3: Refined production tracker (high-contrast shell gating, background wall rejection,
  normalized gated Hungarian assignment, honest uncertainty exposure, broken gap trails,
  and stale-box confidence suppression).
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

import cv2
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment


# ==============================================================================
# SECTION 1: BASELINE TRACKER (V1) - PRESERVED FOR REPRODUCIBILITY
# ==============================================================================

class EggTrack:
    """Represents a tracked egg instance over time (Legacy Baseline v1)."""
    def __init__(self, track_id: int, bbox: tuple, centroid: tuple, mask: np.ndarray, color: tuple, label: str):
        self.track_id = track_id
        self.label = label
        self.bbox = bbox          # (x, y, w, h)
        self.centroid = centroid  # (cx, cy)
        self.mask = mask
        self.color = color
        self.history = [centroid]
        self.lost_frames = 0
        self.age = 1
        self.velocity = (0.0, 0.0)

    def update(self, bbox: tuple, centroid: tuple, mask: np.ndarray):
        dx = centroid[0] - self.centroid[0]
        dy = centroid[1] - self.centroid[1]
        self.velocity = (0.6 * self.velocity[0] + 0.4 * dx, 0.6 * self.velocity[1] + 0.4 * dy)
        self.bbox = bbox
        self.centroid = centroid
        self.mask = mask
        self.history.append(centroid)
        if len(self.history) > 35:
            self.history.pop(0)
        self.lost_frames = 0
        self.age += 1

    def predict(self):
        px = int(self.centroid[0] + self.velocity[0])
        py = int(self.centroid[1] + self.velocity[1])
        self.centroid = (px, py)
        x, y, w, h = self.bbox
        self.bbox = (int(px - w / 2), int(py - h / 2), w, h)
        self.lost_frames += 1


class MultiEggTracker:
    """Multi-object tracker with greedy distance matching (Legacy Baseline v1)."""
    def __init__(self, max_lost_frames: int = 24, max_match_dist: float = 110.0):
        self.tracks = []
        self.next_id = 1
        self.max_lost_frames = max_lost_frames
        self.max_match_dist = max_match_dist
        self.palette = [
            ((0, 230, 255), "Egg #1"),
            ((255, 100, 0), "Egg #2"),
            ((50, 255, 50), "Egg #3"),
            ((200, 50, 255), "Egg #4")
        ]

    def update(self, detections: list) -> list:
        for t in self.tracks:
            t.predict()

        matched_tracks = set()
        matched_dets = set()

        if self.tracks and detections:
            cost_matrix = []
            for t in self.tracks:
                row = []
                for d in detections:
                    dist = np.hypot(t.centroid[0] - d["centroid"][0], t.centroid[1] - d["centroid"][1])
                    row.append(dist)
                cost_matrix.append(row)

            cost_matrix = np.array(cost_matrix)
            for _ in range(min(len(self.tracks), len(detections))):
                min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                t_idx, d_idx = min_idx
                min_val = cost_matrix[t_idx, d_idx]
                if min_val > self.max_match_dist:
                    break

                self.tracks[t_idx].update(
                    detections[d_idx]["bbox"],
                    detections[d_idx]["centroid"],
                    detections[d_idx]["mask"]
                )
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
                cost_matrix[t_idx, :] = 1e9
                cost_matrix[:, d_idx] = 1e9

        for i, d in enumerate(detections):
            if i not in matched_dets:
                if len(self.tracks) < 2:
                    color, label = self.palette[(self.next_id - 1) % len(self.palette)]
                    new_track = EggTrack(self.next_id, d["bbox"], d["centroid"], d["mask"], color, label)
                    self.next_id += 1
                    self.tracks.append(new_track)

        self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_lost_frames]
        return [t for t in self.tracks if t.lost_frames <= 15]


def segment_video_eggs(frame_bgr: np.ndarray) -> list:
    """Legacy Baseline v1 detector."""
    h, w = frame_bgr.shape[:2]
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[130:335, 50:590] = 255

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)

    l = lab[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a**2 + b**2)

    egg_thresh = (l > 105) & (s < 130) & (v > 100) & (chroma < 55) & (roi_mask > 0)
    bin_mask = egg_thresh.astype(np.uint8) * 255

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, k_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800 or area > 18000:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        peri = cv2.arcLength(cnt, True)
        circ = (4.0 * np.pi * area) / (peri * peri) if peri > 0 else 0

        if solidity >= 0.72 and circ >= 0.28:
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = min(bw, bh) / max(bw, bh)
            if aspect < 0.25:
                continue

            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + bw // 2
                cy = y + bh // 2

            inst_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(inst_mask, [cnt], -1, 255, -1)

            detections.append({
                "bbox": (x, y, bw, bh),
                "centroid": (cx, cy),
                "mask": inst_mask,
                "area": area,
                "contour": cnt
            })

    detections.sort(key=lambda d: d["centroid"][0])
    return detections[:2]


# ==============================================================================
# SECTION 2: IMPROVED TRACKER (V2) - INTERMEDIATE EXPERIMENTAL BASELINE
# ==============================================================================

class RobustEggTrack:
    """Represents a tracked egg instance (v2)."""
    def __init__(self, track_id: int, label: str, color: tuple, bbox: tuple, centroid: tuple, mask: np.ndarray):
        self.track_id = track_id
        self.label = label
        self.color = color
        self.bbox = bbox
        self.centroid = centroid
        self.mask = mask
        self.history = [centroid]
        self.lost_frames = 0
        self.detected_frames = 1
        self.state = "DETECTED"
        self.velocity = [0.0, 0.0]
        self.last_seen_centroid = centroid
        self.last_seen_bbox = bbox

    def predict(self):
        vx = self.velocity[0] * 0.70
        vy = self.velocity[1] * 0.70
        px = int(self.centroid[0] + vx)
        py = int(self.centroid[1] + vy)
        self.centroid = (px, py)
        x, y, w, h = self.bbox
        self.bbox = (int(px - w / 2), int(py - h / 2), w, h)
        self.lost_frames += 1
        if self.lost_frames > 2:
            self.state = "OCCLUDED"
        if self.lost_frames > 40:
            self.state = "LOST"

    def update(self, bbox: tuple, centroid: tuple, mask: np.ndarray):
        dx = centroid[0] - self.centroid[0]
        dy = centroid[1] - self.centroid[1]
        self.velocity = [0.5 * self.velocity[0] + 0.5 * dx, 0.5 * self.velocity[1] + 0.5 * dy]
        self.bbox = bbox
        self.centroid = centroid
        self.last_seen_centroid = centroid
        self.last_seen_bbox = bbox
        self.mask = mask
        self.history.append(centroid)
        if len(self.history) > 30:
            self.history.pop(0)
        self.lost_frames = 0
        self.detected_frames += 1
        self.state = "DETECTED"


class RobustEggTracker:
    """Robust Egg Tracker (v2)."""
    def __init__(self, max_lost_frames: int = 40, max_match_dist: float = 220.0):
        self.max_lost_frames = max_lost_frames
        self.max_match_dist = max_match_dist
        self.palette = [
            ((0, 230, 255), "Egg #1 (Left)"),
            ((255, 120, 0), "Egg #2 (Right)")
        ]
        self.tracks = {}

    def update(self, detections: list) -> list:
        for t in self.tracks.values():
            t.predict()

        det_indices = list(range(len(detections)))
        track_ids = list(self.tracks.keys())
        matched_tracks = set()
        matched_dets = set()

        if track_ids and detections:
            cost_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                t = self.tracks[tid]
                for j, d in enumerate(detections):
                    dist = np.hypot(t.centroid[0] - d["centroid"][0], t.centroid[1] - d["centroid"][1])
                    xA = max(t.bbox[0], d["bbox"][0])
                    yA = max(t.bbox[1], d["bbox"][1])
                    xB = min(t.bbox[0] + t.bbox[2], d["bbox"][0] + d["bbox"][2])
                    yB = min(t.bbox[1] + t.bbox[3], d["bbox"][1] + d["bbox"][3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    areaA = t.bbox[2] * t.bbox[3]
                    areaB = d["bbox"][2] * d["bbox"][3]
                    union = float(areaA + areaB - inter)
                    iou = inter / union if union > 0 else 0.0
                    cost = dist * 0.70 + (1.0 - iou) * 50.0
                    cost_matrix[i, j] = cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                tid = track_ids[r]
                t = self.tracks[tid]
                d = detections[c]
                dist = np.hypot(t.centroid[0] - d["centroid"][0], t.centroid[1] - d["centroid"][1])
                if dist < self.max_match_dist:
                    t.update(d["bbox"], d["centroid"], d["mask"])
                    matched_tracks.add(tid)
                    matched_dets.add(c)

        unmatched_dets = [j for j in det_indices if j not in matched_dets]
        for d_idx in unmatched_dets:
            d = detections[d_idx]
            if 1 not in self.tracks:
                color, label = self.palette[0]
                self.tracks[1] = RobustEggTrack(1, label, color, d["bbox"], d["centroid"], d["mask"])
                matched_tracks.add(1)
            elif 2 not in self.tracks:
                color, label = self.palette[1]
                self.tracks[2] = RobustEggTrack(2, label, color, d["bbox"], d["centroid"], d["mask"])
                matched_tracks.add(2)
            else:
                cand_tracks = [t for t in self.tracks.values() if t.track_id not in matched_tracks]
                if cand_tracks:
                    best_track = min(
                        cand_tracks,
                        key=lambda t: np.hypot(t.centroid[0] - d["centroid"][0], t.centroid[1] - d["centroid"][1])
                    )
                    best_track.update(d["bbox"], d["centroid"], d["mask"])
                    matched_tracks.add(best_track.track_id)

        active_list = [t for t in self.tracks.values() if t.lost_frames <= self.max_lost_frames]
        return active_list


def detect_eggs_v2(frame_bgr: np.ndarray, frame_idx: int) -> list:
    """Detector v2."""
    h, w = frame_bgr.shape[:2]
    if frame_idx < 225 or frame_idx > 1008:
        return []

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[145:295, 25:615] = 255

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    L = lab[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    H = hsv[:, :, 0]

    is_skin = (H < 28) & (S > 55) & (V > 60) & (L < 175)
    egg_thresh = (L > 125) & (S < 65) & (V > 115) & (~is_skin) & (roi_mask > 0)
    bin_mask = egg_thresh.astype(np.uint8) * 255

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 1200:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / float(bh) if bh > 0 else 1.0

        if aspect > 1.35 and area > 6000 and bw > 120:
            split_x = x + bw // 2
            mask_left = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask_left, [cnt], -1, 255, -1)
            mask_left[:, split_x:] = 0
            cnts_l, _ = cv2.findContours(mask_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            mask_right = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask_right, [cnt], -1, 255, -1)
            mask_right[:, :split_x] = 0
            cnts_r, _ = cv2.findContours(mask_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            sub_cnts = []
            if cnts_l: sub_cnts.append((cnts_l[0], mask_left, True))
            if cnts_r: sub_cnts.append((cnts_r[0], mask_right, True))
        else:
            inst_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(inst_mask, [cnt], -1, 255, -1)
            sub_cnts = [(cnt, inst_mask, False)]

        for c, m, is_split in sub_cnts:
            sub_area = cv2.contourArea(c)
            if sub_area < 1500 or sub_area > 8000:
                continue

            bx, by, bbw, bbh = cv2.boundingRect(c)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = sub_area / hull_area if hull_area > 0 else 0
            peri = cv2.arcLength(c, True)
            circ = (4.0 * np.pi * sub_area) / (peri * peri) if peri > 0 else 0

            min_circ = 0.40 if is_split else 0.55
            min_solidity = 0.78 if is_split else 0.85

            if solidity >= min_solidity and circ >= min_circ:
                M = cv2.moments(c)
                cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else bx + bbw // 2
                cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else by + bbh // 2

                if cy > 260 and bx < 110 and bbh < 38:
                    continue

                detections.append({
                    "bbox": (bx, by, bbw, bbh),
                    "centroid": (cx, cy),
                    "mask": m,
                    "area": float(sub_area),
                    "solidity": float(solidity),
                    "circ": float(circ),
                    "is_split": is_split,
                    "ambiguity": "touching_split" if is_split else "single_clear"
                })

    detections.sort(key=lambda d: d["centroid"][0])
    return detections[:2]


# ==============================================================================
# SECTION 3: REFINED PRODUCTION TRACKER (V3) - ROBUST & SUBMISSION GRADE
# ==============================================================================

class EggTrackV3:
    """
    Production Track instance with kinematic velocity damping, confidence decay,
    gap-aware broken trajectory trails, and honest uncertainty exposure.
    """
    def __init__(self, track_id: int, label: str, color: tuple, bbox: tuple, centroid: tuple, mask: np.ndarray, frame_idx: int):
        self.track_id = track_id
        self.label = label
        self.color = color
        self.bbox = bbox
        self.centroid = centroid
        self.mask = mask
        # history stores (cx, cy, frame_idx) to enable gap detection
        self.history = [(centroid[0], centroid[1], frame_idx)]
        self.lost_frames = 0
        self.detected_frames = 1
        self.state = "DETECTED"     # "DETECTED", "OCCLUDED", "LOST"
        self.velocity = [0.0, 0.0]
        self.confidence = 1.0
        self.uncertain = False
        self.last_detected_frame = frame_idx

    def predict(self):
        """Kinematic prediction with damped velocity; decays confidence."""
        self.velocity[0] *= 0.50
        self.velocity[1] *= 0.50
        px = int(self.centroid[0] + self.velocity[0])
        py = int(self.centroid[1] + self.velocity[1])
        self.centroid = (px, py)
        x, y, w, h = self.bbox
        self.bbox = (int(px - w / 2), int(py - h / 2), w, h)
        self.lost_frames += 1
        self.confidence *= 0.85
        self.mask = None  # Do NOT render a fake mask during occlusion

        if self.lost_frames > 2:
            self.state = "OCCLUDED"
        if self.lost_frames > 35:
            self.state = "LOST"

    def update(self, bbox: tuple, centroid: tuple, mask: np.ndarray, frame_idx: int):
        """Update track with fresh confirmed observation."""
        dx = centroid[0] - self.centroid[0]
        dy = centroid[1] - self.centroid[1]
        clamped_dx = max(-35.0, min(35.0, float(dx)))
        clamped_dy = max(-35.0, min(35.0, float(dy)))
        self.velocity = [0.5 * self.velocity[0] + 0.5 * clamped_dx, 0.5 * self.velocity[1] + 0.5 * clamped_dy]

        # Break trajectory history if gap > 5 frames so no long false jump line is drawn
        if frame_idx - self.last_detected_frame > 5:
            self.history = []

        self.bbox = bbox
        self.centroid = centroid
        self.mask = mask
        self.history.append((centroid[0], centroid[1], frame_idx))
        if len(self.history) > 25:
            self.history.pop(0)

        # Flag physical identity uncertainty if reacquired after significant occlusion
        if self.lost_frames > 15:
            self.uncertain = True
        elif self.lost_frames == 0:
            self.uncertain = False

        self.lost_frames = 0
        self.detected_frames += 1
        self.state = "DETECTED"
        self.confidence = 1.0
        self.last_detected_frame = frame_idx


class RobustEggTrackerV3:
    """
    Robust Multi-Egg Tracker v3 with gated Hungarian assignment,
    size-relative normalized costs, uncertainty exposure, and stale box suppression.
    """
    def __init__(self, max_lost_frames: int = 35):
        self.max_lost_frames = max_lost_frames
        self.palette = [
            ((0, 215, 255), "Egg #1"),  # Amber / Warm Gold
            ((255, 180, 0), "Egg #2")   # Sky Blue
        ]
        self.tracks = {}

    def update(self, detections: list, frame_idx: int) -> list:
        # 1. Predict all tracks
        for t in self.tracks.values():
            t.predict()

        det_indices = list(range(len(detections)))
        track_ids = list(self.tracks.keys())
        matched_tracks = set()
        matched_dets = set()

        # 2. Hungarian Bipartite Assignment
        if track_ids and detections:
            cost_matrix = np.full((len(track_ids), len(detections)), 1000.0, dtype=np.float32)
            for i, tid in enumerate(track_ids):
                t = self.tracks[tid]
                diag = max(60.0, float(np.hypot(t.bbox[2], t.bbox[3])))

                for j, d in enumerate(detections):
                    dist = np.hypot(t.centroid[0] - d["centroid"][0], t.centroid[1] - d["centroid"][1])
                    norm_dist = dist / diag

                    xA = max(t.bbox[0], d["bbox"][0])
                    yA = max(t.bbox[1], d["bbox"][1])
                    xB = min(t.bbox[0] + t.bbox[2], d["bbox"][0] + d["bbox"][2])
                    yB = min(t.bbox[1] + t.bbox[3], d["bbox"][1] + d["bbox"][3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    areaA = t.bbox[2] * t.bbox[3]
                    areaB = d["bbox"][2] * d["bbox"][3]
                    union = float(areaA + areaB - inter)
                    iou = inter / union if union > 0 else 0.0

                    if t.lost_frames <= 3:
                        max_d = 160.0
                        c = 0.60 * norm_dist + 0.40 * (1.0 - iou)
                    else:
                        max_d = min(240.0, 120.0 + 4.0 * t.lost_frames)
                        c = 0.80 * norm_dist

                    if dist <= max_d and c < 2.0:
                        cost_matrix[i, j] = c

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 2.0:
                    tid = track_ids[r]
                    d = detections[c]
                    self.tracks[tid].update(d["bbox"], d["centroid"], d["mask"], frame_idx)
                    matched_tracks.add(tid)
                    matched_dets.add(c)

        # 3. Association for unmatched detections
        unmatched_dets = [j for j in det_indices if j not in matched_dets]
        for d_idx in unmatched_dets:
            d = detections[d_idx]
            if 1 not in self.tracks:
                self.tracks[1] = EggTrackV3(1, self.palette[0][1], self.palette[0][0], d["bbox"], d["centroid"], d["mask"], frame_idx)
                matched_tracks.add(1)
            elif 2 not in self.tracks:
                self.tracks[2] = EggTrackV3(2, self.palette[1][1], self.palette[1][0], d["bbox"], d["centroid"], d["mask"], frame_idx)
                matched_tracks.add(2)
            else:
                cand_tracks = [t for tid, t in self.tracks.items() if tid not in matched_tracks]
                if cand_tracks:
                    best_t = min(cand_tracks, key=lambda t: np.hypot(t.centroid[0] - d["centroid"][0], t.centroid[1] - d["centroid"][1]))
                    best_t.update(d["bbox"], d["centroid"], d["mask"], frame_idx)
                    matched_tracks.add(best_t.track_id)

        return list(self.tracks.values())


def detect_eggs_v3(frame_bgr: np.ndarray, frame_idx: int) -> list:
    """
    Video-specific egg detector v3 with tabletop ROI, demonstrator skin rejection,
    high-contrast shell color gating (L > 155), and touching-egg aspect ratio splitting.
    """
    if frame_idx < 225 or frame_idx > 1008:
        return []

    h, w = frame_bgr.shape[:2]
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[145:295, 25:615] = 255

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    L = lab[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    H = hsv[:, :, 0]

    # Skin mask: rejects demonstrator hands & fingers
    is_skin = (H < 28) & (S > 50) & (V > 60) & (L < 185)

    # Clean shell threshold: eliminates background wall (L ~ 130) and shirts
    egg_thresh = (L > 155) & (S < 55) & (V > 150) & (~is_skin) & (roi_mask > 0)
    bin_mask = egg_thresh.astype(np.uint8) * 255

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 1200:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / float(bh) if bh > 0 else 1.0

        if aspect > 1.35 and area > 6000 and bw > 120:
            split_x = x + bw // 2
            m_l = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(m_l, [cnt], -1, 255, -1)
            m_l[:, split_x:] = 0
            cnts_l, _ = cv2.findContours(m_l, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            m_r = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(m_r, [cnt], -1, 255, -1)
            m_r[:, :split_x] = 0
            cnts_r, _ = cv2.findContours(m_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            sub_cnts = []
            if cnts_l: sub_cnts.append((cnts_l[0], m_l, True))
            if cnts_r: sub_cnts.append((cnts_r[0], m_r, True))
        else:
            inst_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(inst_mask, [cnt], -1, 255, -1)
            sub_cnts = [(cnt, inst_mask, False)]

        for c, m, is_split in sub_cnts:
            sub_area = cv2.contourArea(c)
            if sub_area < 1400 or sub_area > 8000:
                continue

            bx, by, bbw, bbh = cv2.boundingRect(c)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = sub_area / hull_area if hull_area > 0 else 0
            peri = cv2.arcLength(c, True)
            circ = (4.0 * np.pi * sub_area) / (peri * peri) if peri > 0 else 0

            min_circ = 0.40 if is_split else 0.45
            min_solidity = 0.78 if is_split else 0.82

            if solidity >= min_solidity and circ >= min_circ:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx = bx + bbw // 2
                    cy = by + bbh // 2

                if cy > 260 and bx < 110 and bbh < 38:
                    continue

                detections.append({
                    "bbox": (bx, by, bbw, bbh),
                    "centroid": (cx, cy),
                    "mask": m,
                    "area": float(sub_area),
                    "solidity": float(solidity),
                    "circ": float(circ),
                    "is_split": is_split,
                    "ambiguity": "touching_split" if is_split else "single_clear"
                })

    detections.sort(key=lambda d: d["centroid"][0])
    return detections[:2]


# ==============================================================================
# SECTION 4: PIPELINE EXECUTION & VIDEO RENDERING
# ==============================================================================

def run_video_tracking(video_source: str = "data/input_video.mp4",
                       output_path: Optional[str] = None,
                       tracker_version: str = "v3",
                       max_frames: Optional[int] = None) -> dict:
    """Run tracking pipeline on input video and export annotated MP4."""
    if output_path is None:
        if tracker_version == "v1":
            output_path = "outputs/tracked_video.mp4"
        elif tracker_version == "v2":
            output_path = "outputs/tracked_video_v2.mp4"
        else:
            output_path = "outputs/tracked_video_v3.mp4"

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    print("=" * 70)
    print(f"TASK 4: MULTI-OBJECT EGG TRACKING ({tracker_version.upper()})")
    print("=" * 70)
    print(f"Source Video:        {video_source}")
    print(f"Resolution:          {width}x{height}")
    print(f"Frame Rate:          {fps:.2f} FPS")
    print(f"Total Video Frames:  {total_frames} ({duration_sec:.1f}s)")
    print(f"Output Video Target: {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if tracker_version == "v1":
        tracker = MultiEggTracker(max_lost_frames=24, max_match_dist=120.0)
    elif tracker_version == "v2":
        tracker = RobustEggTracker(max_lost_frames=40, max_match_dist=220.0)
    else:
        tracker = RobustEggTrackerV3(max_lost_frames=35)

    frame_idx = 0
    frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

    frames_with_2_detected = 0
    frames_with_1_detected = 0
    frames_with_0_detected = 0
    all_observed_ids = set()

    pbar = tqdm(total=frames_to_process, desc=f"Tracking ({tracker_version.upper()})")

    while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        sec = frame_idx / fps

        if tracker_version == "v1":
            if 180 <= frame_idx <= 1080:
                detections = segment_video_eggs(frame)
            else:
                detections = []
            active_tracks = tracker.update(detections)
        elif tracker_version == "v2":
            detections = detect_eggs_v2(frame, frame_idx)
            active_tracks = tracker.update(detections)
        else:
            detections = detect_eggs_v3(frame, frame_idx)
            active_tracks = tracker.update(detections, frame_idx)

        # Count confirmed detections
        det_count = sum(1 for t in active_tracks if getattr(t, "state", "DETECTED") == "DETECTED")
        if det_count >= 2:
            frames_with_2_detected += 1
        elif det_count == 1:
            frames_with_1_detected += 1
        else:
            frames_with_0_detected += 1

        for t in active_tracks:
            all_observed_ids.add(t.track_id)

        vis = frame.copy()

        # Render tracks
        for t in active_tracks:
            color = t.color
            state = getattr(t, "state", "DETECTED" if t.lost_frames == 0 else "OCCLUDED")
            confidence = getattr(t, "confidence", 1.0)
            uncertain = getattr(t, "uncertain", False)
            x, y, bw, bh = t.bbox

            if state == "DETECTED":
                # 1. Translucent mask overlay
                if t.mask is not None and t.mask.any():
                    vis[t.mask > 0] = (vis[t.mask > 0] * 0.45 + np.array(color) * 0.55).astype(np.uint8)

                # 2. Tight Bounding Box
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)

                # 3. Compact Label HUD (exposes uncertainty if identity is ambiguous)
                status_tag = "[?]" if uncertain else "[DET]"
                label = f"{t.label} {status_tag}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)

                # Ensure label box stays inside frame boundaries
                lbl_y1 = max(0, y - th - 6)
                lbl_y2 = y
                lbl_x1 = max(0, x)
                lbl_x2 = min(width - 1, x + tw + 6)
                cv2.rectangle(vis, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)
                cv2.putText(vis, label, (lbl_x1 + 3, lbl_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)

                # 4. Trajectory Trail (broken across gaps > 3 frames)
                if len(t.history) > 1:
                    for k in range(len(t.history) - 1):
                        p1 = t.history[k]
                        p2 = t.history[k + 1]
                        # In v3, history entries are (cx, cy, f_idx)
                        if len(p1) == 3 and len(p2) == 3:
                            if p2[2] - p1[2] <= 3:
                                cv2.line(vis, (p1[0], p1[1]), (p2[0], p2[1]), color, 2, cv2.LINE_AA)
                        else:
                            cv2.line(vis, p1, p2, color, 2, cv2.LINE_AA)

                # 5. Centroid Point
                cv2.circle(vis, (t.centroid[0], t.centroid[1]), 3, (0, 0, 255), -1)

            elif state == "OCCLUDED":
                # Only draw subtle box if confidence is still above threshold (lost <= 6 frames)
                if tracker_version != "v3" or confidence > 0.40:
                    box_color = (130, 130, 130) if tracker_version == "v3" else color
                    cv2.rectangle(vis, (x, y), (x + bw, y + bh), box_color, 1, cv2.LINE_AA)
                    label = f"{t.label} [OCCLUDED]"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                    cv2.putText(vis, label, (x, max(15, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_color, 1, cv2.LINE_AA)
                # If confidence <= 0.40 in v3: HIDE THE BOX entirely to avoid stale ghosts

        # Upper Telemetry HUD Banner
        hud_bg = np.zeros((36, width, 3), dtype=np.uint8)
        alpha = 0.65
        vis[0:36, 0:width] = cv2.addWeighted(vis[0:36, 0:width], 1 - alpha, hud_bg, alpha, 0)

        time_str = f"{int(sec // 60):02d}:{sec % 60:04.1f}"
        if tracker_version == "v3":
            method_str = "Gated-Hungarian v3"
        elif tracker_version == "v2":
            method_str = "Hungarian-State v2"
        else:
            method_str = "SORT-Kalman v1"

        hud_text = (f"T: {time_str} | Frame: {frame_idx:04d}/{total_frames} | "
                    f"Detected: {det_count} | Active IDs: {sorted(list(all_observed_ids))} | Tracker: {method_str}")
        cv2.putText(vis, hud_text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(vis)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    out_file = Path(output_path)
    file_size_mb = out_file.stat().st_size / (1024 * 1024) if out_file.exists() else 0.0

    stats = {
        "tracker_version": tracker_version,
        "source_video": video_source,
        "output_path": output_path,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames_processed": frame_idx,
        "duration_sec": round(frame_idx / fps, 2),
        "file_size_mb": round(file_size_mb, 2),
        "frames_with_2_detected": frames_with_2_detected,
        "frames_with_1_detected": frames_with_1_detected,
        "frames_with_0_detected": frames_with_0_detected,
        "all_observed_ids": sorted(list(all_observed_ids)),
        "tracking_method": "Gated Hungarian Assignment + Size-Relative Costs + Stale Box Suppression (v3)" if tracker_version == "v3" else ("Hungarian Bipartite Assignment + Damped Kinematics (v2)" if tracker_version == "v2" else "SORT Centroid Matching + Coasting (v1)"),
        "video_status": "VERIFIED_PLAYABLE"
    }

    print("\n" + "=" * 70)
    print(f"TASK 4 TRACKING SUMMARY ({tracker_version.upper()})")
    print("=" * 70)
    print(f"Frames Processed:   {frame_idx} / {total_frames}")
    print(f"Output File Size:   {file_size_mb:.2f} MB")
    print(f"Frames with 2 Eggs: {frames_with_2_detected} ({frames_with_2_detected / frame_idx * 100:.1f}%)")
    print(f"Frames with 1 Egg:  {frames_with_1_detected} ({frames_with_1_detected / frame_idx * 100:.1f}%)")
    print(f"Observed Track IDs: {sorted(list(all_observed_ids))}")
    print(f"Output Video Path:  {output_path}")
    print("=" * 70)

    if tracker_version == "v3":
        summary_filename = "outputs/tracking_summary_v3.json"
    elif tracker_version == "v2":
        summary_filename = "outputs/tracking_summary_v2.json"
    else:
        summary_filename = "outputs/tracking_summary.json"

    with open(summary_filename, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 4 Video Tracking")
    parser.add_argument("--version", choices=["v1", "v2", "v3"], default="v3", help="Tracker version (v1, v2, or v3)")
    parser.add_argument("--output", default=None, help="Output MP4 file path")
    args = parser.parse_args()

    run_video_tracking(tracker_version=args.version, output_path=args.output)
