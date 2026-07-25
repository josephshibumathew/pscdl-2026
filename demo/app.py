"""
PSCDL 2026 — Live Demo (Gradio Interface)
Run with: gradio app.py
"""

import os
import cv2
import numpy as np
import gradio as gr
import tempfile
import shutil

# Import your pipeline
import sys
sys.path.append("..")  # Go up one level to find src/
from src.generate_mask import generate_mask
from src.pipeline import run_pipeline


def process_video(video_file):
    """
    Process uploaded video and return visualization.
    """
    # 1. Save uploaded video to temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(video_file)
    temp_video.close()

    video_path = temp_video.name

    # 2. Run the pipeline (only first 120 seconds for demo speed)
    result = run_pipeline(
        video_path=video_path,
        p=60,
        c=90,
        min_area=1500,
        max_area=60000,
        diff_threshold=25,
        recent_ratio=0.70,
    )

    masks = result['masks']
    duration = result['duration']
    h, w = result['height'], result['width']

    # 3. Find the active window (where detection happens)
    # We want to show a frame where the object is detected.
    # Look for the first mask that has white pixels.
    active_sec = None
    for sec, mask in enumerate(masks):
        if mask.max() > 0:
            active_sec = sec
            break

    if active_sec is None:
        return None, "No persistent object detected in this video."

    # 4. Get original frame at that second
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(active_sec * fps + fps / 2))
    ret, frame = cap.read()
    cap.release()

    # 5. Create overlay: original frame + green mask overlay
    overlay = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = masks[active_sec]

    # Convert mask to 3-channel for overlay
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_3ch[mask > 127] = [0, 255, 0]  # Green overlay

    # Blend: 70% original, 30% mask
    overlay = cv2.addWeighted(overlay, 0.7, mask_3ch, 0.3, 0)

    # 6. Cleanup
    os.remove(video_path)

    # 7. Return visualizations
    return (
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),  # Original frame
        mask,                                      # Mask
        overlay,                                   # Overlay
        f"Detected at second {active_sec} / {duration}s",
    )


# ============================================================
# Gradio Interface
# ============================================================

with gr.Blocks(
    title="PSCDL 2026 - Persistent Scene Change Detection",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown("""
    # 🏆 PSCDL 2026 — Persistent Scene Change Detection
    ### 1st Place F1 Score (0.40) on Blind Test

    **How it works:**
    1. Upload a fixed-camera surveillance video (MP4).
    2. The system detects objects left behind for ≥ 60 seconds.
    3. It flags them for exactly 30 seconds (between 60s and 90s).
    4. Green overlay shows detected persistent objects.

    **Try it with:**
    - A video of a bag left on a street.
    - Or a parked motorcycle.
    """)

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(
                label="Upload Surveillance Video",
                interactive=True,
                height=400,
            )
            submit_btn = gr.Button("🔍 Detect Persistent Objects", variant="primary")

        with gr.Column():
            output_original = gr.Image(label="📷 Original Frame", height=300)
            output_mask = gr.Image(label="🎯 Predicted Mask", height=300)
            output_overlay = gr.Image(label="🟢 Detection Overlay", height=300)
            output_info = gr.Textbox(label="📊 Detection Info")

    # Connect button to function
    submit_btn.click(
        fn=process_video,
        inputs=video_input,
        outputs=[
            output_original,
            output_mask,
            output_overlay,
            output_info,
        ],
    )

    # Example videos (if you have a sample video hosted somewhere)
    gr.Markdown("""
    ---
    **📌 Note:** This demo processes the video and finds the first detected persistent object.
    For best results, upload a video where an object appears and stays for at least 60 seconds.
    """)

    with gr.Row():
        gr.Examples(
            examples=[
                # Add your own example video paths here if you have them
                # ["sample_video.mp4"],
            ],
            inputs=video_input,
            fn=process_video,
            outputs=[
                output_original,
                output_mask,
                output_overlay,
                output_info,
            ],
            cache_examples=False,
        )


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":
    demo.launch(share=True)