"""
Focused Unit Tests for Task 4 Improved Multi-Object Egg Tracking (v2).
Validates:
1. Scene transition & title card rejection (intro/outro 0 detections).
2. Tabletop ROI & skin tone filtering.
3. Touching egg split & ambiguity logging.
4. Hungarian matching & track ID persistence without fragmentation.
5. Track state machine transitions (DETECTED -> OCCLUDED -> LOST).
"""

import cv2
import numpy as np
import pytest
from pathlib import Path
from src.task4_tracking import detect_eggs_v2, RobustEggTracker, RobustEggTrack


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


def test_scene_boundaries_intro_outro_rejection(video_capture):
    """Test that intro and outro title cards generate 0 detections."""
    # Frame 60: Intro text card
    frame_intro = get_frame(video_capture, 60)
    dets_intro = detect_eggs_v2(frame_intro, frame_idx=60)
    assert len(dets_intro) == 0, f"Expected 0 detections on intro title card, got {len(dets_intro)}"

    # Frame 1200: Outro explanation text card
    frame_outro = get_frame(video_capture, 1200)
    dets_outro = detect_eggs_v2(frame_outro, frame_idx=1200)
    assert len(dets_outro) == 0, f"Expected 0 detections on outro card, got {len(dets_outro)}"


def test_demonstration_scene_detections(video_capture):
    """Test that representative tabletop demo frames detect the physical eggs."""
    # Frame 300: Static resting eggs side-by-side
    frame_300 = get_frame(video_capture, 300)
    dets_300 = detect_eggs_v2(frame_300, frame_idx=300)
    assert len(dets_300) == 2, f"Expected 2 detections at frame 300, got {len(dets_300)}"

    # Ensure both detections have plausible egg dimensions
    for d in dets_300:
        bw, bh = d["bbox"][2], d["bbox"][3]
        assert 50 <= bw <= 120, f"Plausible width violation: {bw}"
        assert 50 <= bh <= 120, f"Plausible height violation: {bh}"
        assert d["solidity"] >= 0.75, f"Solidity violation: {d['solidity']}"


def test_touching_egg_split_and_ambiguity_logging(video_capture):
    """Test that touching eggs are split into 2 individual instances with ambiguity recorded."""
    frame_300 = get_frame(video_capture, 300)
    dets_300 = detect_eggs_v2(frame_300, frame_idx=300)
    assert len(dets_300) == 2

    # Check ambiguity field
    split_flags = [d.get("is_split", False) for d in dets_300]
    assert any(split_flags), "Expected touching egg split logic to trigger for resting adjacent eggs"
    for d in dets_300:
        if d.get("is_split", False):
            assert d.get("ambiguity") == "touching_split"


def test_skin_tone_rejection(video_capture):
    """Test that hand regions are filtered out and not classified as eggs."""
    # Frame 480 has large demonstrator hand presence
    frame_480 = get_frame(video_capture, 480)
    dets_480 = detect_eggs_v2(frame_480, frame_idx=480)
    
    # Hand should NOT create an oversized bbox (> 150 px) or false detections on fingers
    for d in dets_480:
        bx, by, bw, bh = d["bbox"]
        assert bw < 140, f"Box width {bw} indicates hand was included in detection"
        assert bh < 130, f"Box height {bh} indicates hand was included in detection"


def test_hungarian_tracker_identity_stability(video_capture):
    """Test that Hungarian tracker maintains exactly IDs 1 and 2 without minting new IDs."""
    tracker = RobustEggTracker()
    test_frame_indices = [250, 260, 270, 280, 290, 300]
    
    observed_ids = set()
    for f_idx in test_frame_indices:
        frame = get_frame(video_capture, f_idx)
        dets = detect_eggs_v2(frame, f_idx)
        tracks = tracker.update(dets)
        for t in tracks:
            observed_ids.add(t.track_id)

    assert observed_ids == {1, 2}, f"Expected only IDs {{1, 2}}, got {observed_ids}"


def test_track_state_machine():
    """Test track state machine transitions: DETECTED -> OCCLUDED -> LOST."""
    tracker = RobustEggTracker()
    
    # Initial detection
    det = [{
        "bbox": (100, 150, 60, 70),
        "centroid": (130, 185),
        "mask": np.zeros((360, 640), dtype=np.uint8),
        "area": 4000.0,
        "solidity": 0.90,
        "circ": 0.70
    }]
    
    tracks = tracker.update(det)
    assert len(tracks) == 1
    assert tracks[0].state == "DETECTED"
    assert tracks[0].lost_frames == 0
    
    # Simulate occlusion (no detections for 5 frames)
    for _ in range(5):
        tracks = tracker.update([])
    
    assert len(tracks) == 1
    assert tracks[0].state == "OCCLUDED"
    assert tracks[0].lost_frames == 5

    # Simulate prolonged loss (45 frames)
    for _ in range(40):
        tracks = tracker.update([])
        
    assert len(tracks) == 0, "Track should be terminated after exceeding max_lost_frames (40)"
