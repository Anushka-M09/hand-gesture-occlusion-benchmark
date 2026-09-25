# Robustness Benchmarking of Hand Gesture Recognition Under Partial Occlusion

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-brightgreen.svg)](https://developers.google.com/mediapipe)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, production-ready computer vision and machine learning framework designed to evaluate the failure modes and degradation thresholds of landmark-based hand gesture recognition under contiguous physical occlusions.

---

## 🎯 Core Objectives & Research Motivation

In real-world human-computer interaction (HCI) and robotic perception, hand gestures frequently suffer from **partial occlusion** (e.g., hands holding objects, sleeves, microphone sticks, or camera framing cuts). 

Standard vision pipelines rely on intermediate keypoint estimators (such as **MediaPipe Hands**). This project systematically measures:
1. **Landmark Detector Failure Thresholds:** At what occlusion percentage ($0\%, 10\%, 20\%, 30\%, 45\%, 60\%$) does MediaPipe fail to detect landmark topologies?
2. **Classifier Robustness:** When landmarks *are* partially extracted under occlusion, how well does the downstream machine learning classifier (Random Forest, MLP, SVM) maintain gesture discrimination?
3. **Generalization via Invariant Landmark Normalization:** Eliminates sensitivity to hand size, position in frame, and distance from camera.

---

## 📐 Mathematical Feature Normalization

Raw MediaPipe landmarks provide 21 coordinates $(x_i, y_i, z_i)$ in image-relative space. To achieve **full translation and scale invariance**, each detected hand undergoes the following mathematical transformation:

1. **Translation Invariance (Wrist Centering):**
   Subtract the wrist landmark $\vec{p}_0 = (x_0, y_0, z_0)$ from all 21 points:
   $$\vec{p}'_i = \vec{p}_i - \vec{p}_0 \quad \forall i \in \{0, 1, \dots, 20\}$$
   *Result:* Landmark 0 (wrist) is always located exactly at $(0, 0, 0)$.

2. **Scale Invariance (Radial Normalization):**
   Compute the maximum Euclidean distance from the wrist to any keypoint/fingertip:
   $$d_{\max} = \max_{i \in \{0, \dots, 20\}} \|\vec{p}'_i\|_2$$
   Scale all coordinates:
   $$\vec{p}''_i = \frac{\vec{p}'_i}{d_{\max} + \epsilon}$$
   *Result:* All landmark coordinates are strictly bounded within $[-1.0, 1.0]$, independent of hand size or camera distance.

3. **Vector Flattening:**
   The normalized coordinates are flattened into a 1D feature vector of length **63** ($21 \times 3$).

---

## 📂 Directory Structure

```text
hand_gesture_occlusion_benchmark/
│
├── data/
│   ├── raw/                 # Clean gesture images organized by class subfolders
│   │   ├── fist/
│   │   ├── ok/
│   │   ├── peace/
│   │   ├── pointing/
│   │   ├── thumbs_down/
│   │   └── thumbs_up/
│   └── occluded/            # Generated occluded images across benchmark levels
│
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py      # Automated streaming downloader for public datasets (HaGRID)
│   ├── dataset_collector.py # Interactive OpenCV webcam recording utility & mock generator
│   ├── occluder.py          # Synthetic contiguous block/cutout occlusion generator
│   ├── feature_extractor.py # MediaPipe Hands wrapper with invariant normalization
│   ├── model.py             # Classifier training (RandomForest / MLP / SVM) & export
│   ├── benchmark.py         # Systematic stress-testing loop over occlusion levels
│   └── visualize.py         # Plots: Accuracy curves, drop-off rates, confusion matrices
│
├── outputs/                 # Benchmark CSV logs and publication-ready figures
├── models/                  # Serialized classifier bundle & test split manifest
├── main.py                  # CLI pipeline runner connecting all modules
├── requirements.txt         # Core dependencies
└── README.md                # Project documentation
```

---

## ⚡ Installation & Setup

1. **Clone or navigate to the repository:**
   ```bash
   cd "hand_gesture_occlusion_benchmark"
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Pipeline

`main.py` provides an intuitive command-line interface for each phase of the workflow:

### Option A: Download Public Dataset (HaGRID subset)
Automate data collection with diverse public hands from Hugging Face:
```bash
python main.py --fetch --samples 200
```

### Option B: Record Custom Gestures with Webcam
Open the interactive OpenCV tool to record ~50 samples per class with a 3-second countdown:
```bash
python main.py --collect --samples 50
```
*Webcam Controls:*
- `[SPACE]` : Begin 3-second countdown to record a 20-frame burst.
- `[N]` : Cycle to next gesture class.
- `[P]` : Cycle to previous gesture class.
- `[Q]` : Save and quit.

### Option C: Instant Synthetic Mock Dataset (No Camera / Offline Testing)
Generate synthetic hand gesture images immediately to test the entire pipeline:
```bash
python main.py --mock-data --samples 40
```

---

### Step 2: Train the Classifier on Clean Images (0% Occlusion)
Extract 63-dimensional normalized landmarks, split into 80/20 train/test sets, and train the model:
```bash
# Default: Random Forest
python main.py --train --model-type random_forest

# Or train Multi-Layer Perceptron (MLP) or Support Vector Machine (SVM):
python main.py --train --model-type mlp
```
*Output:* Serialized bundle saved to `models/gesture_classifier.joblib` and test split to `models/test_split.json`.

---

### Step 3: Run the Occlusion Stress Test
Evaluate held-out test frames against synthetic block occlusions across `[0%, 10%, 20%, 30%, 45%, 60%]`:
```bash
python main.py --benchmark
```
*Metrics track:*
- MediaPipe detection rate (% frames where landmarks were successfully recovered).
- Overall classification accuracy (% correct out of all test frames).
- Conditional accuracy (% correct among frames where detector succeeded).

---

### Step 4: Generate Visualizations & Failure Analysis
Render high-resolution figures in `outputs/`:
```bash
python main.py --visualize
```

Generated charts:
1. `outputs/accuracy_vs_occlusion.png` : Line plot of overall accuracy and per-class curves vs. occlusion %.
2. `outputs/detection_dropoff.png` : MediaPipe detector failure curve highlighting the critical breakdown threshold.
3. `outputs/confusion_matrices.png` : Side-by-side heatmaps showing misclassification shifts at 0%, 30%, and 60% occlusion.
4. `outputs/occlusion_visual_grid.png` : Montage grid showing sample images across the occlusion spectrum.

---

### Step 5: Full Pipeline End-to-End
To run data preparation, model training, benchmarking, and visualization in a single command:
```bash
python main.py --all
```

---

## 🔬 Occlusion Simulation Details

Unlike simple salt-and-pepper noise, real physical obstructions (such as hands obstructed by coffee mugs or sleeves) are **contiguous**. 

`src/occluder.py` calculates the hand region-of-interest (ROI) using landmark bounding boxes and places contiguous rectangular blocks simulating:
- **Solid black blocks** (simulating opaque physical blockage).
- **Random textured noise blocks** (simulating irregular surface occlusions).
- **Gaussian blur cutouts** (simulating depth-of-field or sensor smudge).

The area of the block corresponds strictly to the percentage parameter:
$$\text{Area}_{\text{block}} = \text{occlusion\_pct} \times (\text{Width}_{\text{hand}} \times \text{Height}_{\text{hand}})$$

---

## 📊 Summary Output Format

The benchmark logs every individual trial to `outputs/benchmark_results.csv`:
| image_path | ground_truth | occlusion_pct | detection_success | predicted | is_correct | confidence |
|---|---|---|---|---|---|---|
| data/raw/thumbs_up/001.jpg | thumbs_up | 0.0 | True | thumbs_up | True | 0.98 |
| data/raw/thumbs_up/001.jpg | thumbs_up | 0.30 | True | thumbs_up | True | 0.84 |
| data/raw/thumbs_up/001.jpg | thumbs_up | 0.60 | False | UNDETECTED | False | 0.00 |

Aggregated metrics are saved to `outputs/benchmark_summary.csv`.
