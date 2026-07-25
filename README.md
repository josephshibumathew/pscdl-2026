# 🏆 PSCDL 2026 Winner — Persistent Scene Change Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green)
![Competition](https://img.shields.io/badge/PSCDL-2026-1st_Place-gold)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
[![CI / Tests](https://github.com/josephshibumathew/pscdl-2026-solution/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/pscdl-2026-solution/actions/workflows/ci.yml)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-yellow)](https://huggingface.co/spaces/Josephsmathew/pscdl-2026-demo)
> **A classical computer vision pipeline for detecting left-behind objects in fixed-camera surveillance video.**

<br>

## 📈 Competition Results (Blind Test Set)
| Metric | Rank | Score |
|--------|------|-------|
| 🥇 **Overall Winner** | **1st** | 0.4 on Blind Test |
| 🥇 **F1 Score** | **1st** | **0.40** |
| 🥉 **Inference Time** | **3rd** | Optimized |

<br>

## 🧠 Key Innovation: Circular Buffer Architecture
Storing the last 90 seconds of frames naively takes **746 MB**. Storing the whole video takes **3.4 GB**. 
This pipeline implements a **circular buffer (uint8)** that updates in **O(1) time** and uses a constant **186 MB** of memory, independent of video length.

<br>

## 📝 Problem Statement
- **Task**: Generate a binary mask per second (White = persistent object, Black = background).
- **p/c Rule**: Object must exist for `p=60s` before flagging. Flag for `(c-p)=30s`, then stop (`c=90s`).
- **Challenge**: Reject transient objects (pedestrians, shadows) while handling occlusions.

<br>

## ⚙️ Pipeline Architecture
Video → Median Background (8-15s)
→ ROI & Noise Masking (exclude sky/trees)
→ Frame Differencing (Threshold = 25)
→ Circular Buffer (90 frames, 186MB)
→ Persistence Check (window_c ≥ 60)
→ Recency Filter (70% presence in last 60s)
→ Seed Expansion & Blob Filter (1500-60000 px)
→ Binary Masks (mask_XXXX.png)

text

<br>

## 📦 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/pscdl-2026-solution.git
cd pscdl-2026-solution
2. Install dependencies

bash
pip install -r requirements.txt
3. Run inference on a video

python
from generate_mask import generate_mask

# p = 60 seconds, c = 90 seconds
generate_mask(p=60, c=90, video_path="path/to/your/video.mp4")
Output: Masks are saved to output_masks/mask_0001.png, mask_0002.png, ...


📂 Directory Structure

text
├── generate_mask.py          # Main pipeline code
├── requirements.txt          # Dependencies
├── README.md                 # You are here!
└── output_masks/             # Generated binary masks

🔍 Detailed Approach

1. Background Modeling

Uses Median of the first 8–15 seconds (grayscale). Robust to pedestrian outliers (Mean would create "ghosts").
Adaptive duration: min(15, max(8, 10% of duration)).
2. ROI & Noise Masking

Excludes top 20% (sky/trees) and bottom timestamp.
Auto Noise Detection: Pixels noisy in >50% of clean frames are permanently masked out (structural noise).
3. Frame Differencing

Samples one frame per second (midpoint). Threshold = 25 (empirically tuned).
Morphological cleanup: MORPH_OPEN (removes noise) → MORPH_CLOSE (fills holes).
4. Persistence Logic (The Core)

Sliding Window: window_c tracks how many of the last 90 seconds a pixel was white.
first_active Map: Records the exact second a pixel first hits count >= 60.
Flagging Window: Flags exactly for (c-p) = 30 seconds.
Recency Filter: Requires 70% presence in the last 60 seconds. Rejects pedestrian corridors (20% presence) while keeping stationary objects (95% presence).
5. Seed Expansion & Blob Filtering

Dilates persistent seeds (15x15) to capture the full object (especially hollow centers).
Clips expansion to the raw difference map to prevent bleeding into the background.
Data-Driven Blob Filter: Noise blobs measured ≤1521px, True objects ≥4091px. Set min_area=1500, max_area=60000.

📊 Development Set Performance

Video	Object	Precision	Recall	F1
video_1	Trolley bags	0.163	0.735	0.267
video_2	Small bag (≈892px)	0.000	0.000	0.000*
video_3	Motorcycle	0.828	0.692	0.754
video_4	Thela cart	0.130	0.655	0.217
video_5	Multiple objects	0.104	0.318	0.157
*video_2 F1=0 is an accepted trade-off (object below min_area). Lowering the threshold allowed 1521px noise blobs into other videos, destroying overall precision.


🏁 Next Steps / Future Work

SAM2 Refinement: Use coarse detections as prompts for precise boundary segmentation.
Occlusion-Robust Tracking: Implement Kalman filters to maintain object IDs through long occlusions.
Adaptive Thresholding: Per-video calibration from the first 5 clean seconds to catch tiny objects without reintroducing noise.

📜 License

This project is open-source and available under the MIT License.


🙏 Acknowledgements

PSCDL 2026 / NCVPRIPG 2026 for organizing the challenge.
Vehant Technologies for providing the dataset and evaluation framework.
Government Model Engineering College, Kochi for the support.

Made with ❤️ by Jiyaro Joseph & Joseph S. Mathew
