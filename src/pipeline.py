"""
Core pipeline logic for PSCDL 2026.
"""

import cv2
import numpy as np
import gc

from src.background import build_background, get_adaptive_background_duration
from src.region import get_detection_region


def run_pipeline(
    video_path: str,
    p: int = 60,
    c: int = 90,
    min_area: int = 1500,
    max_area: int = 60000,
    diff_threshold: int = 25,
    recent_ratio: float = 0.70,
) -> dict:
    """
    Run the persistent change detection pipeline on a video.

    Args:
        video_path: Path to input video
        p: Persistence threshold (seconds)
        c: Cooldown period (seconds)
        min_area: Minimum blob area to keep
        max_area: Maximum blob area to keep
        diff_threshold: Threshold for frame differencing
        recent_ratio: Recency filter ratio (0.0-1.0)

    Returns:
        Dictionary with: masks (list of 2D arrays), duration, fps, h, w
    """
    # Read video properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    cap.release()

    print(f"  {video_path.split('/')[-1]} | {duration}s | {fps:.1f}fps")

    # Build background and detection region
    bg_secs = get_adaptive_background_duration(duration)
    background = build_background(video_path, fps, bg_secs)
    clean_secs = min(bg_secs, int(duration * 0.10))
    det_region = get_detection_region(h, w, background, video_path, fps, clean_secs)

    # Derived parameters
    recent_threshold = int(p * recent_ratio)
    flagging_window = c - p
    masks_start = p + 10

    # Kernels
    smooth_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    exp_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # Circular buffer (c x h x w, uint8) — 186MB for c=90
    buf = np.zeros((c, h, w), dtype=np.uint8)
    window_c = np.zeros((h, w), dtype=np.int32)
    first_active = np.full((h, w), -1, dtype=np.int32)

    masks = []
    cap = cv2.VideoCapture(video_path)

    for sec in range(duration):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps + fps / 2))
        ok, frame = cap.read()

        if ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, background)
            diff = cv2.bitwise_and(diff, diff, mask=det_region)

            _, cur = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
            cur = cv2.morphologyEx(cur, cv2.MORPH_OPEN, smooth_k)
            cur = cv2.morphologyEx(cur, cv2.MORPH_CLOSE, smooth_k)
            cur_bin = (cur > 0).astype(np.uint8)
        else:
            cur_bin = np.zeros((h, w), dtype=np.uint8)
            diff = np.zeros((h, w), dtype=np.uint8)

        # Circular buffer update
        slot = sec % c
        window_c -= buf[slot].astype(np.int32)
        buf[slot] = cur_bin
        window_c += cur_bin.astype(np.int32)

        # Record first activation
        if sec >= masks_start:
            first_active[(window_c >= p) & (first_active < 0)] = sec

        # Generate mask
        if sec < masks_start:
            masks.append(np.zeros((h, w), dtype=np.uint8))
            continue

        elapsed = sec - first_active
        seeds = (
            (first_active >= 0) &
            (elapsed >= 0) &
            (elapsed < flagging_window)
        ).astype(np.uint8) * 255

        if seeds.max() == 0:
            masks.append(np.zeros((h, w), dtype=np.uint8))
            continue

        # Recency filter
        recent = np.zeros((h, w), dtype=np.int32)
        for i in range(p):
            recent += buf[(sec - i) % c].astype(np.int32)

        seeds = cv2.bitwise_and(
            seeds,
            (recent >= recent_threshold).astype(np.uint8) * 255
        )

        if seeds.max() == 0:
            masks.append(np.zeros((h, w), dtype=np.uint8))
            continue

        # Seed expansion and clipping
        seeds = cv2.morphologyEx(seeds, cv2.MORPH_CLOSE, close_k)
        seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, open_k)
        expanded = cv2.dilate(seeds, exp_k)

        _, diff_bin = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
        result = cv2.bitwise_and(expanded, diff_bin)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, close_k)

        # Blob area filtering
        n, labels, stats, _ = cv2.connectedComponentsWithStats(result)
        final = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                final[labels == i] = 255

        masks.append(final)

    cap.release()

    # Cleanup
    del buf, window_c, first_active, background
    gc.collect()

    return {
        'masks': masks,
        'duration': duration,
        'fps': fps,
        'height': h,
        'width': w,
    }