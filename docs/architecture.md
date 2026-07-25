# PSCDL 2026 — Architecture Overview

## High-Level Summary
A classical computer vision pipeline for persistent scene change detection in fixed-camera surveillance video.

**Core Innovation**: A circular buffer sliding window that reduces memory from **3.4GB to 186MB** (constant).

**Competition Results**: #1 F1 (0.40) on blind test, #3 Inference Time.

---

## Pipeline Stages

### 1. Background Modeling (Median)
- **Input**: First 8-15 seconds of grayscale frames.
- **Method**: `np.median(frames, axis=0)`
- **Why Median**: Robust to pedestrians (appear in <10% of frames). Mean would create ghost artifacts.
- **Adaptive Duration**: `min(15, max(8, 10% of video_duration))`

### 2. ROI & Noise Masking
- **Hard Exclusions**: 
  - Top 20% (sky, trees — only swaying noise)
  - Bottom 45px (burned-in timestamp)
- **Automatic Structural Noise Detection**:
  - Sample clean period frames.
  - Mark pixels noisy in >50% of samples.
  - Remove only large blobs (≥5000px) to avoid masking object locations.

### 3. Frame Differencing
- Sample one frame per second (midpoint to avoid motion blur).
- `diff = cv2.absdiff(gray, background)`
- Threshold = 25 (empirically tuned; values below 25 are sensor noise).
- **Morphological Cleanup**:
  - `MORPH_OPEN`: Remove small salt-and-pepper noise.
  - `MORPH_CLOSE`: Fill holes inside objects.

### 4. Circular Buffer (Core Innovation)
- **Problem**: Storing 90 frames naively costs 746MB (3.4GB for a full video).
- **Solution**: Fixed-size buffer of `c=90` slots (`uint8` dtype).
- **Update Rule**:
```python
slot = sec % 90
window_c -= buf[slot]   # remove oldest (90s ago)
buf[slot] = cur_bin     # overwrite with today's frame
window_c += cur_bin     # add newest to running sum
Memory: 90 × 1080 × 1920 × 1 = 186MB (constant).
Time: O(1) update per second.
5. Persistence Logic & Flagging

window_c = how many of the last 90 seconds the pixel was white.
first_active = exact second when window_c first reaches >= p (60).
Flagging Window: elapsed = sec - first_active. Flag only if 0 <= elapsed < (c - p) = 30.
Critical Fix: Originally flagged for c seconds (90s). Corrected to c-p (30s) to match ground truth.
6. Recency Filter (Pedestrian Rejection)

Sum the last p=60 frames from the circular buffer.
Require presence in ≥70% of recent 60 seconds.
Stationary Objects: 95% recency → pass.
Pedestrian Corridors: 20-25% recency → rejected.
7. Seed Expansion & Blob Filtering

Dilate seeds by 15px to fill the hollow center of low-contrast objects.
Clip expansion to diff_bin (prevents bleeding into background).
Blob Area Filter (Data-Driven):

False positives measured: ≤ 1521px.
True objects measured: ≥ 4091px.
min_area = 1500, max_area = 60000.
Key Parameters

Parameter	Value	Justification
diff_threshold	25	Above sensor noise, below real object edges
min_area	1500px	Sits in 2570px gap between noise and objects
max_area	60000px	Removes road texture false positives
recent_ratio	0.70	Separates stationary (95%) from pedestrians (20%)
bg_secs	8-15s	Adaptive to video length
expand_px	15px	Fills hollow centers without over-expanding
Performance

Metric	Value
Blind Test F1	0.40 (1st Place)
Inference Time	3rd Place
Memory	186MB (constant)
Development F1	0.244 (overall)
Directory Structure

text
src/                  # Core package
├── background.py     # Median background + adaptive duration
├── region.py         # ROI + structural noise masking
├── pipeline.py       # Circular buffer + persistence logic
└── generate_mask.py  # Main entry point + CLI

scripts/              # Utility scripts
├── run_dev_set.py    # Process 5 videos + compute F1
├── evaluate_f1.py    # Standalone F1 evaluation
└── submit_test.py    # Test set + zip submission

tests/                # Unit tests (pytest)
└── test_pipeline.py

demo/                 # Gradio live demo (Hugging Face)
└── app.py
Future Work

SAM2 Refinement: Use coarse seeds as prompts for precise object boundaries.
Slow Background Update: Running average to handle day/night cycles (currently fixed).
Image Stabilization: Remove camera jitter (wind vibration).