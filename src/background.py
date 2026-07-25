"""
Background modeling utilities for PSCDL 2026.
"""

import cv2
import numpy as np


def build_background(video_path, fps, seconds=15):
    """
    Build a median background model from the first N seconds of video.

    Args:
        video_path: Path to input video
        fps: Frames per second of the video
        seconds: Number of seconds to use for background modeling

    Returns:
        2D grayscale image (uint8) of the median background
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    for _ in range(int(fps * seconds)):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    cap.release()

    if not frames:
        raise ValueError(f"No frames read from {video_path}")

    return np.median(np.array(frames), axis=0).astype(np.uint8)


def get_adaptive_background_duration(duration):
    """
    Compute adaptive background duration: 10% of video, clamped between 8-15s.

    Args:
        duration: Total video duration in seconds

    Returns:
        Optimal background duration in seconds
    """
    return min(15, max(8, int(duration * 0.10)))