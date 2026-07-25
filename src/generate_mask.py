"""
PSCDL 2026 - Persistent Scene Change Detection

Usage:
    python -m src.generate_mask --p 60 --c 90 --video_path video.mp4

Or import:
    from src.generate_mask import generate_mask
    generate_mask(p=60, c=90, video_path="video.mp4")
"""

import os
import argparse
import shutil

from src.pipeline import run_pipeline


def generate_mask(p: int, c: int, video_path: str) -> None:
    """
    Generate per-second binary masks for persistent scene change detection.

    Args:
        p: Persistence threshold (seconds) — object must be present this long
        c: Cooldown period (seconds) — stop flagging after this long
        video_path: Path to input video file

    Output:
        Saves masks to output_masks/mask_0001.png, mask_0002.png, ...
    """
    # Clean output directory
    if os.path.exists("output_masks"):
        shutil.rmtree("output_masks")
    os.makedirs("output_masks")

    # Run pipeline
    result = run_pipeline(
        video_path=video_path,
        p=p,
        c=c,
        min_area=1500,
        max_area=60000,
        diff_threshold=25,
        recent_ratio=0.70,
    )

    # Save masks
    for sec, mask in enumerate(result['masks']):
        path = f"output_masks/mask_{str(sec + 1).zfill(4)}.png"
        cv2.imwrite(path, mask)

    saved = len(result['masks'])
    print(f"  Saved {saved} masks ({'✓' if saved == result['duration'] else '✗ MISMATCH'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PSCDL 2026 - Persistent Scene Change Detection"
    )
    parser.add_argument("--p", type=int, default=60,
                        help="Persistence threshold in seconds")
    parser.add_argument("--c", type=int, default=90,
                        help="Cooldown period in seconds")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to input video file")

    args = parser.parse_args()
    generate_mask(p=args.p, c=args.c, video_path=args.video_path)