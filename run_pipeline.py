"""
Master Pipeline Orchestrator for Computer Vision Intern Assessment.
Executes all required phases sequentially while respecting incremental caches:
- Stage 1: Pretrained Foundation Segmentation (SAM ViT-B Pseudo-GT)
- Stage 2: Classical Segmentation Trials & Experiment Log
- Stage 3: Quantitative & Qualitative Evaluation vs SAM Pseudo-GT
- Stage 4: Multi-Object Video Tracking (YouTube Physics Demonstration)
- Stage 5: Publication-Grade PDF Report Generation
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Ensure root in path
sys.path.insert(0, os.path.abspath("."))


def run_pipeline(skip_sam: bool = False, skip_tracking: bool = False):
    total_t0 = time.time()
    print("=" * 75)
    print("CV INTERN ASSESSMENT: MASTER PIPELINE ORCHESTRATOR")
    print("=" * 75)
    print(f"Python Executable: {sys.executable}")
    print(f"Working Directory: {os.path.abspath('.')}\n")

    # STAGE 1: Pretrained Foundation Segmentation
    print("\n" + "=" * 75)
    print(">>> STAGE 1 / 5: PRETRAINED SEGMENTATION (SAM ViT-B)")
    print("=" * 75)
    if skip_sam:
        print("Skipping Stage 1 (--skip-sam flag specified).")
    else:
        from src.task1_pretrained import main as run_task1
        # Task 1 internally checks for existing metadata and masks, skipping precomputed outputs
        run_task1(config_path="configs/config.yaml")

    # STAGE 2: Classical Segmentation Trials
    print("\n" + "=" * 75)
    print(">>> STAGE 2 / 5: CLASSICAL SEGMENTATION & EXPERIMENT LOG")
    print("=" * 75)
    from src.task2_classical import run_trials as run_task2
    run_task2(config_path="configs/config.yaml")

    # STAGE 3: Evaluation & Diagnostics
    print("\n" + "=" * 75)
    print(">>> STAGE 3 / 5: EVALUATION & QUALITATIVE DIAGNOSTICS")
    print("=" * 75)
    from src.task3_evaluation import evaluate_system as run_task3
    run_task3(config_path="configs/config.yaml")

    # STAGE 4: Video Tracking
    print("\n" + "=" * 75)
    print(">>> STAGE 4 / 5: MULTI-OBJECT EGG TRACKING")
    print("=" * 75)
    if skip_tracking:
        print("Skipping Stage 4 (--skip-tracking flag specified).")
    else:
        from src.task4_tracking import run_video_tracking as run_task4
        run_task4()

    # STAGE 5: PDF Report Compilation
    print("\n" + "=" * 75)
    print(">>> STAGE 5 / 5: COMPILING FORMAL PDF ASSESSMENT REPORT")
    print("=" * 75)
    from src.generate_pdf_report import build_pdf_report as run_task5
    report_file = run_task5()

    elapsed = time.time() - total_t0
    print("\n" + "=" * 75)
    print(f"ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {elapsed:.2f}s ({elapsed/60:.2f}m)!")
    print(f"Final PDF Submission: {os.path.abspath(report_file)}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Assessment Pipeline Runner")
    parser.add_argument("--skip-sam", action="store_true", help="Skip SAM ViT-B inference if masks exist")
    parser.add_argument("--skip-tracking", action="store_true", help="Skip video tracking if tracked MP4 exists")
    args = parser.parse_args()

    run_pipeline(skip_sam=args.skip_sam, skip_tracking=args.skip_tracking)
