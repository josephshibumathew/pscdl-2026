"""
Run the pipeline on test videos and create submission zip.
"""

import os
import sys
import cv2
import zipfile
import shutil
import gc
import argparse

# Add parent directory to path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generate_mask import generate_mask


def copy_masks_clean(src_dir, dest_dir):
    """
    Clean destination first, then copy masks.

    Args:
        src_dir: Source directory with masks
        dest_dir: Destination directory
    """
    # Delete old files in destination
    if os.path.exists(dest_dir):
        for f in os.listdir(dest_dir):
            os.remove(os.path.join(dest_dir, f))
    else:
        os.makedirs(dest_dir)

    # Copy fresh
    for f in sorted(os.listdir(src_dir)):
        shutil.copy(f"{src_dir}/{f}", f"{dest_dir}/{f}")

    return len(os.listdir(dest_dir))


def process_test_videos(test_dir, output_dir=None, p=60, c=90, zip_path=None):
    """
    Process all test videos and optionally create submission zip.

    Args:
        test_dir: Directory containing test videos (.mp4 files)
        output_dir: Directory to save masks (default: same as test_dir)
        p: Persistence threshold
        c: Cooldown period
        zip_path: Path for output zip file (optional)

    Returns:
        List of processed video names
    """
    # Find all test videos
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".mp4")])

    if not test_files:
        print("No test videos found!")
        return []

    print(f"Found {len(test_files)} test videos: {test_files}")

    if output_dir is None:
        output_dir = test_dir

    processed_videos = []

    for filename in test_files:
        video_path = os.path.join(test_dir, filename)
        video_name = filename.replace(".mp4", "")
        video_mask_dir = os.path.join(output_dir, f"{video_name}_output_masks")

        print(f"\nProcessing: {filename}")

        # Run pipeline
        generate_mask(p=p, c=c, video_path=video_path)

        # Copy masks to named folder
        mask_count = copy_masks_clean("output_masks", video_mask_dir)
        print(f"  Copied {mask_count} masks → {video_mask_dir}")

        processed_videos.append(video_name)

        # Cleanup
        gc.collect()

    # Create zip if requested
    if zip_path:
        print(f"\nCreating zip: {zip_path}")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for video_name in processed_videos:
                video_mask_dir = os.path.join(output_dir, f"{video_name}_output_masks")
                if os.path.exists(video_mask_dir):
                    for mask_file in sorted(os.listdir(video_mask_dir)):
                        full_path = os.path.join(video_mask_dir, mask_file)
                        arcname = os.path.join(video_name, mask_file)
                        zf.write(full_path, arcname)

        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"  Zip created: {zip_path} ({size_mb:.1f} MB)")

        # Verify zip
        with zipfile.ZipFile(zip_path, 'r') as zf_check:
            print(f"  Zip contains {len(zf_check.namelist())} files")

    return processed_videos


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PSCDL 2026 - Test Set Submission Generator"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Directory containing test videos (.mp4 files)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save masks (default: same as test_dir)",
    )
    parser.add_argument(
        "--zip_path",
        type=str,
        help="Path for output zip file (e.g., submission.zip)",
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

    print("=" * 60)
    print("PSCDL 2026 — Test Set Submission Generator")
    print("=" * 60)
    print(f"Test directory: {args.test_dir}")
    print(f"p={args.p}, c={args.c}")
    print("=" * 60)

    process_test_videos(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        p=args.p,
        c=args.c,
        zip_path=args.zip_path,
    )

    print("\n" + "=" * 60)
    print("Done! Submit the zip file to the challenge organizers.")


if __name__ == "__main__":
    main()