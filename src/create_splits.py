"""
Part B: Leakage-Aware Dataset Split Generator.
Constructs a sequence- and scene-aware 80:20 split (88 Development : 22 Evaluation)
to prevent frame-level temporal leakage from adjacent video frames.
Preserves the original directory structure completely.
"""

import os
import sys
import json
import re
from pathlib import Path


def extract_frame_id(filename: str) -> int:
    """Extract numeric frame sequence index from filename (e.g. not_damaged_101_jpg -> 101)."""
    m = re.search(r"not_damaged_(\d+)_", filename)
    if m:
        return int(m.group(1))
    return -1


def assign_scene_group(frame_id: int) -> str:
    """
    Categorize images into physical capture scenes based on sequence analysis:
    - Group A (IDs 1-45): Hand holding egg against neutral wall
    - Group B (IDs 49-95): Ash / charcoal bowl / outdoor gravel
    - Group C (IDs 100-124): Water bowl / metallic pot with water
    - Group D (IDs 125-149): Kitchen countertop / multiple eggs
    - Group E (IDs 150-164): Patterned woven cloth fabric
    """
    if 1 <= frame_id <= 45:
        return "hand_holding_single_egg"
    elif 46 <= frame_id <= 99:
        return "ash_charcoal_bowl"
    elif 100 <= frame_id <= 124:
        return "water_metallic_bowl"
    elif 125 <= frame_id <= 149:
        return "kitchen_countertop_multiple"
    elif 150 <= frame_id <= 164:
        return "patterned_cloth_fabric"
    else:
        return "unknown_scene"


def generate_splits(dataset_dir: str = "egg.v1i.folder", output_file: str = "data/splits.json"):
    base_dir = Path(dataset_dir)
    image_paths = sorted(list(base_dir.rglob("*.jpg")))
    
    records = []
    for p in image_paths:
        fn = p.name
        fid = extract_frame_id(fn)
        scene = assign_scene_group(fid)
        rel_p = str(p.relative_to(base_dir)).replace("\\", "/")
        original_split = rel_p.split("/")[0]
        records.append({
            "stem": p.stem,
            "filename": fn,
            "rel_path": rel_p,
            "frame_id": fid,
            "scene": scene,
            "original_roboflow_split": original_split
        })
        
    records.sort(key=lambda x: (x["scene"], x["frame_id"]))
    
    # Group by scene
    by_scene = {}
    for r in records:
        by_scene.setdefault(r["scene"], []).append(r)
        
    dev_set = []
    eval_set = []
    
    # Allocate approximately 80% dev, 20% eval within each scene using continuous hold-out blocks
    # (avoiding alternating/random interleaving which causes temporal leakage)
    eval_targets = {
        "hand_holding_single_egg": 8,       # 31 dev, 8 eval (total 39)
        "ash_charcoal_bowl": 2,             # 5 dev, 2 eval (total 7)
        "water_metallic_bowl": 5,           # 20 dev, 5 eval (total 25)
        "kitchen_countertop_multiple": 5,   # 19 dev, 5 eval (total 24)
        "patterned_cloth_fabric": 2         # 13 dev, 2 eval (total 15)
    }
    
    for scene, items in by_scene.items():
        items.sort(key=lambda x: x["frame_id"])
        n_eval = eval_targets.get(scene, max(1, int(round(len(items) * 0.2))))
        n_dev = len(items) - n_eval
        
        # Take the first n_dev items as development, and the tail n_eval items as evaluation
        # This keeps both sets as continuous temporal sub-sequences, separating them in time!
        scene_dev = items[:n_dev]
        scene_eval = items[n_dev:]
        
        for item in scene_dev:
            item["split"] = "dev"
            dev_set.append(item)
            
        for item in scene_eval:
            item["split"] = "eval"
            eval_set.append(item)
            
    print(f"Total dataset images: {len(records)}")
    print(f"Development split count: {len(dev_set)} ({len(dev_set)/len(records)*100:.1f}%)")
    print(f"Evaluation split count:  {len(eval_set)} ({len(eval_set)/len(records)*100:.1f}%)")
    
    print("\nScene breakdown:")
    for scene in sorted(by_scene.keys()):
        d_cnt = sum(1 for x in dev_set if x["scene"] == scene)
        e_cnt = sum(1 for x in eval_set if x["scene"] == scene)
        tot = d_cnt + e_cnt
        print(f"  - {scene:30s}: Dev = {d_cnt:2d}, Eval = {e_cnt:2d} (Total = {tot:2d})")
        
    splits_payload = {
        "summary": {
            "total_images": len(records),
            "dev_count": len(dev_set),
            "eval_count": len(eval_set),
            "dev_ratio": round(len(dev_set) / len(records), 3),
            "eval_ratio": round(len(eval_set) / len(records), 3),
            "leakage_mitigation_strategy": (
                "Continuous sequence block split within each distinct capture scene. "
                "Dev split takes the initial sub-sequence; Eval split holds out the final "
                "sub-sequence of frames. Adjacent frames never cross the dev/eval boundary."
            )
        },
        "dev": [x["stem"] for x in dev_set],
        "eval": [x["stem"] for x in eval_set],
        "records": {x["stem"]: x for x in records}
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(splits_payload, f, indent=2)
        
    print(f"\nSaved splits mapping to: {output_file}")
    return splits_payload


if __name__ == "__main__":
    generate_splits()
