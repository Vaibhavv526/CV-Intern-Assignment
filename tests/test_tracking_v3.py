"""
Regression and Unit Tests for Task 4 Refined Multi-Object Egg Tracker (v3).
Covers:
1. Scene transition & intro/outro title card rejection (0 detections).
2. Bug fix for 24-28s interval: left egg detection at frames 600 and 672 without wall bleed.
3. Stale box suppression when confidence <= 0.40.
4. Prohibition of fake segmentation masks during occlusion (predict sets mask=None).
5. Broken trajectory trails across occlusion gaps (> 5 frames).
6. Uncertainty exposure ([?]) upon reacquisition after prolonged occlusion (> 15 frames).
7. Normalized gated Hungarian assignment (rejects associations with cost >= 2.0).
8. Strict 2-egg ID persistence without track fragmentation.
"""

import os
import sys
import pytest
import numpy as np
import cv2

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.task4_tracking import (
    RobustEggTrackerV3, EggTrackV3, detect_eggs_v3
)


@pytest.fixture
def video_capture():
    cap = cv2.VideoCapture("data/input_video.mp4")
    assert cap.isOpened(), "Could not open data/input_video.mp4"
    yield cap
    cap.release()


def get_frame(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ret, frame = cap.read()
    assert ret, f"Could not read frame {frame_number}"
    return frame


def test_intro_outro_zero_detections(video_capture):
    """Intro and outro title cards must yield 0 detections."""
    frame_intro = get_frame(video_capture, 60)
    dets_intro = detect_eggs_v3(frame_intro, frame_idx=60)
    assert len(dets_intro) == 0, f"Expected 0 detections on intro, got {len(dets_intro)}"

    frame_outro = get_frame(video_capture, 1200)
    dets_outro = detect_eggs_v3(frame_outro, frame_idx=1200)
    assert len(dets_outro) == 0, f"Expected 0 detections on outro, got {len(dets_outro)}"


def test_left_egg_reacquisition_bug_fix(video_capture):
    """
    Regression test for the 24-28s defect:
    In v2, the left egg at frames 560-680 was merged into the background wall
    because of low L threshold (L > 125). In v3 (L > 155), the left egg
    must be cleanly detected.
    """
    # Frame 600 (25.0s)
    frame_600 = get_frame(video_capture, 600)
    dets_600 = detect_eggs_v3(frame_600, frame_idx=600)
    assert len(dets_600) >= 1, "Expected detections at frame 600"
    
    # Verify that an egg on the left side of the table (x < 250) is detected
    left_dets_600 = [d for d in dets_600 if d["centroid"][0] < 250]
    assert len(left_dets_600) == 1, "Left egg must be detected at frame 600"
    left_egg = left_dets_600[0]
    bx, by, bw, bh = left_egg["bbox"]
    assert 40 <= bw <= 110 and 40 <= bh <= 110, f"Invalid bbox dimensions: {left_egg['bbox']}"
    assert left_egg["area"] < 8000, "Left egg must not bleed into background wall"

    # Frame 672 (28.0s)
    frame_672 = get_frame(video_capture, 672)
    dets_672 = detect_eggs_v3(frame_672, frame_idx=672)
    left_dets_672 = [d for d in dets_672 if d["centroid"][0] < 250]
    assert len(left_dets_672) == 1, "Left egg must be detected at frame 672"


def test_stale_box_suppression_and_confidence_decay():
    """
    Test that occluded tracks decay confidence by 0.85 per frame.
    When confidence <= 0.40 (lost > 6 frames), the tracker suppresses the box.
    """
    track = EggTrackV3(1, "Egg #1", (0, 215, 255), (100, 100, 60, 60), (130, 130), np.zeros((10, 10)), 10)
    assert track.confidence == 1.0
    assert track.state == "DETECTED"

    # Frame 1 loss
    track.predict()
    assert track.lost_frames == 1
    assert abs(track.confidence - 0.85) < 1e-4

    # Lost for 7 frames: 0.85^7 = 0.3205 <= 0.40
    for _ in range(6):
        track.predict()
    assert track.lost_frames == 7
    assert track.confidence < 0.40
    assert track.state == "OCCLUDED"


def test_no_fake_mask_during_occlusion():
    """Predicting a track state during occlusion must set mask = None."""
    dummy_mask = np.ones((50, 50), dtype=np.uint8) * 255
    track = EggTrackV3(1, "Egg #1", (0, 215, 255), (100, 100, 60, 60), (130, 130), dummy_mask, 10)
    assert track.mask is not None

    track.predict()
    assert track.mask is None, "Occluded track must NOT retain or synthesize a fake mask"


def test_trajectory_gap_breaking():
    """
    Test that trajectory history breaks across gaps > 5 frames
    so no artificial long line connects a stale position to a new location.
    """
    track = EggTrackV3(1, "Egg #1", (0, 215, 255), (100, 100, 60, 60), (130, 130), None, 10)
    track.update((105, 105, 60, 60), (135, 135), None, frame_idx=11)
    assert len(track.history) == 2

    # Simulate 8 lost frames (gap = 8 > 5)
    for _ in range(8):
        track.predict()

    # Reacquire at frame 20
    track.update((300, 200, 60, 60), (330, 230), None, frame_idx=20)
    
    # Old stale points (130, 130) and (135, 135) must be cleared!
    assert len(track.history) == 1, "History must reset after a gap > 5 frames"
    assert track.history[0] == (330, 230, 20)


def test_uncertainty_flagging_on_long_occlusion():
    """
    If a track was lost for > 15 frames, reacquisition must flag uncertainty (uncertain = True).
    """
    track = EggTrackV3(1, "Egg #1", (0, 215, 255), (100, 100, 60, 60), (130, 130), None, 10)
    assert not track.uncertain

    # Lost for 18 frames
    for _ in range(18):
        track.predict()
    assert track.lost_frames == 18

    # Reacquire
    track.update((110, 110, 60, 60), (140, 140), None, frame_idx=29)
    assert track.uncertain is True, "Track reacquired after > 15 lost frames must be flagged uncertain"


def test_gated_hungarian_rejects_distant_false_matches():
    """
    Hungarian tracker must NOT forcibly match a detection if distance / cost exceeds threshold.
    """
    tracker = RobustEggTrackerV3()
    
    # Initialize track 1 at (100, 100)
    init_det = [{
        "bbox": (70, 70, 60, 60),
        "centroid": (100, 100),
        "mask": np.zeros((10, 10)),
        "area": 3000.0,
        "solidity": 0.90,
        "circ": 0.70
    }]
    tracks = tracker.update(init_det, frame_idx=1)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1

    # Simulate a detection far away at (500, 300)
    distant_det = [{
        "bbox": (470, 270, 60, 60),
        "centroid": (500, 300),
        "mask": np.zeros((10, 10)),
        "area": 3000.0,
        "solidity": 0.90,
        "circ": 0.70
    }]
    
    # Distance is ~447 px > max_d (160 px). Gating must prevent forcibly snapping Track 1!
    tracks = tracker.update(distant_det, frame_idx=2)
    # Track 1 should be occluded/lost, and distant_det should initialize Track 2
    t1 = tracker.tracks[1]
    assert t1.lost_frames == 1, "Track 1 should not have matched distant detection"
    assert 2 in tracker.tracks, "Distant detection should be assigned to Track 2"


def test_id_persistence_two_eggs(video_capture):
    """
    Tracking over normal demonstration frames must preserve strictly IDs {1, 2}.
    """
    tracker = RobustEggTrackerV3()
    observed_ids = set()

    for f_idx in range(240, 320, 10):
        frame = get_frame(video_capture, f_idx)
        dets = detect_eggs_v3(frame, f_idx)
        tracks = tracker.update(dets, f_idx)
        for t in tracks:
            observed_ids.add(t.track_id)

    assert observed_ids == {1, 2}, f"Observed IDs must be exactly {{1, 2}}, got {observed_ids}"
