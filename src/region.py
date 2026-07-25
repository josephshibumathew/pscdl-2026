"""
ROI and noise masking utilities for PSCDL 2026.
"""

import cv2
import numpy as np


def get_detection_region(h, w, background, video_path, fps, clean_duration=12):
    """
    Build a detection region mask that excludes permanently noisy areas.

    Steps:
    1. Exclude top 20% (sky/trees) and bottom 45px (timestamp)
    2. Sample clean period to find structurally noisy pixels
    3. Remove large noisy blobs (≥5000px) from detection region

    Args:
        h, w: Frame dimensions
        background: Background image
        video_path: Path to video
        fps: Frames per second
        clean_duration: Seconds to sample for noise detection

    Returns:
        2D mask (uint8) where 255 = allowed, 0 = excluded
    """
    # Start with all pixels allowed
    region = np.ones((h, w), dtype=np.uint8) * 255

    # Hardcoded exclusions
    region[:int(h * 0.20), :] = 0  # Top 20%: sky/trees
    region[h - 45:, :] = 0  # Bottom 45px: timestamp

    # Auto-detect structural noise
    noise_acc = np.zeros((h, w), dtype=np.int32)
    count = 0
    cap = cv2.VideoCapture(video_path)
    step = max(1, clean_duration // 15)

    for t in range(0, clean_duration, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        noise_acc += (cv2.absdiff(gray, background) > 25).astype(np.int32)
        count += 1

    cap.release()

    if count > 0:
        # Pixels noisy in >50% of sampled frames
        noisy = (noise_acc / count > 0.50).astype(np.uint8) * 255

        # Dilate to merge nearby noisy pixels
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        noisy = cv2.dilate(noisy, k)

        # Find connected components
        n, labels, stats, _ = cv2.connectedComponentsWithStats(noisy)

        # Only exclude large blobs (≥5000px) — preserves object locations
        filtered = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= 5000:
                filtered[labels == i] = 255

        # Remove noisy regions from detection region
        region = cv2.bitwise_and(region, cv2.bitwise_not(filtered))

    return region