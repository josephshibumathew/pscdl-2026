"""
Run the pipeline on all 5 development videos and compute F1 scores.
"""

import os
import sys
import cv2
import numpy as np
import shutil
import gc
import argparse

# Add parent directory to path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generate_mask import generate_mask


# Ground truth intervals for all 5 videos (from txt files)
GT_INTERVALS = {
    1: {
        "mask1.png": (0, 87),
        "mask2.png": (88, 117),
        "mask3.png": (118, 207),
        "mask4.png": (208, 237),
        "mask5.png": (238, 240),
    },
    2: {
        "mask1.png": (0, 82),
        "mask2.png": (83, 112),
        "mask3.png": (113, 120),
    },
    3: {
        "mask1.png": (0, 259),
        "mask2.png": (260, 289),
        "mask3.png": (290, 368),
    },
    4: {
        "mask1.png": (0, 135),
        "mask2.png": (136, 165),
        "mask3.png": (166, 338),
    },
    5: {
        "mask1.png": (0, 152),
        "mask2.png": (153, 175),
        "mask3.png": (176, 182),
        "mask4.png": (183, 205),
        "mask5.png": (206, 243),
        "mask6.png": (244, 259),
        "mask7.png": (260, 273),
        "mask8.png": (274, 289),
        "mask9.png": (290, 417),
    },
}


def compute_f1_for_video(vid_num, base_dir):
    """Compute F1 score for a single video."""
    vid_dir = f"{base_dir}/video_{vid_num}"
    out_dir = f"{vid_dir}/output_masks"

    if not os.path.exists(out_dir):
        print(f"  Warning: {out_dir} not found")
        return 0.0, 0.0, 0.0, 0, 0, 0

    gt_intervals = GT_INTERVALS[vid_num]
    tp = fp = fn = 0

    for gt_fname, (start, end) in gt_intervals.items():
        gt_path = os.path.join(vid_dir, gt_fname)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if gt is None:
            print(f"  Warning: {gt_fname} not found")
            continue

        gt_bin = (gt > 127).astype(np.uint8)

        for sec in range(start, end + 1):
            pred_path = os.path.join(out_dir, f"mask_{str(sec + 1).zfill(4)}.png")

            if os.path.exists(pred_path):
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                pred_bin = (pred > 127).astype(np.uint8)
            else:
                pred_bin = np.zeros_like(gt_bin)

            tp += int(np.sum(pred_bin & gt_bin))
            fp += int(np.sum(pred_bin & (1 - gt_bin)))
            fn += int(np.sum((1 - pred_bin) & gt_bin))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1, tp, fp, fn


def copy_masks_clean(src_dir, dest_dir):
    """Clean destination first, then copy masks."""
    if os.path.exists(dest_dir):
        for f in os.listdir(dest_dir):
            os.remove(os.path.join(dest_dir, f))
    else:
        os.makedirs(dest_dir)

    for f in sorted(os.listdir(src_dir)):
        shutil.copy(f"{src_dir}/{f}", f"{dest_dir}/{f}")

    return len(os.listdir(dest_dir))


def run_dev_set(base_dir, p=60, c=90):
    """Process all 5 videos and print results."""
    print("\n" + "=" * 60)
    print("PSCDL 2026 — Development Set Evaluation")
    print("=" * 60)
    print(f"Base directory: {base_dir}")
    print(f"p={p}, c={c}")
    print("=" * 60)

    total_tp = total_fp = total_fn = 0

    print(f"\n{'Video':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>10} {'FP':>10} {'FN':>10}")
    print("-" * 78)

    for vid_num in range(1, 6):
        video_path = f"{base_dir}/video_{vid_num}/video_{vid_num}.mp4"

        if not os.path.exists(video_path):
            print(f"video_{vid_num} not found, skipping")
            continue

        print(f"\nProcessing video_{vid_num}...")

        # Generate masks
        generate_mask(p=p, c=c, video_path=video_path)

        # Copy to Drive
        dest = f"{base_dir}/video_{vid_num}/output_masks"
        copy_masks_clean("output_masks", dest)

        # Compute F1
        pr, rc, f1, tp, fp, fn = compute_f1_for_video(vid_num, base_dir)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"video_{vid_num}  {pr:>10.4f} {rc:>10.4f} {f1:>10.4f} {tp:>10} {fp:>10} {fn:>10}")

        gc.collect()

    # Overall
    p_all = total_tp / (total_tp + total_fp + 1e-8)
    r_all = total_tp / (total_tp + total_fn + 1e-8)
    f1_all = 2 * p_all * r_all / (p_all + r_all + 1e-8)

    print("-" * 78)
    print(f"{'OVERALL':<8} {p_all:>10.4f} {r_all:>10.4f} {f1_all:>10.4f} {total_tp:>10} {total_fp:>10} {total_fn:>10}")
    print("\n" + "=" * 60)
    print(f"Done! Overall F1: {f1_all:.4f}")
    print("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PSCDL 2026 - Run on development set and compute F1"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/content/drive/MyDrive/PSCDL_2026",
        help="Base directory containing video_1/, video_2/, ...",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=60,
        help="Persistence threshold (default: 60)",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=90,
        help="Cooldown period (default: 90)",
    )

    args = parser.parse_args()
    run_dev_set(base_dir=args.base_dir, p=args.p, c=args.c)


if __name__ == "__main__":
    main()