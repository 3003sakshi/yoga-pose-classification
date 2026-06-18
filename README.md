# 🧘 Yoga Pose Classification & Quality Assessment

A computer-vision system that detects **7 yoga poses** and rates their **execution quality** (Good / Average / Poor) in real-time, using MediaPipe for body-keypoint extraction and a Random Forest classifier.



## Overview

Yoga practitioners often lack real-time feedback on whether they are performing poses correctly. This project bridges that gap by building an end-to-end ML pipeline that:

1. **Extracts** 33-point body keypoints from video frames using [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose).
2. **Engineers** 71 discriminative features (joint angles, limb distances, normalised coordinates).
3. **Classifies** pose identity and execution quality with a tuned Random Forest model.
4. **Provides** correction tips based on the predicted quality level.

The system achieves **~95% accuracy** across 21 classes (7 poses × 3 quality levels).

---

## Demo

Sample results on held-out test videos:

| Video | Predicted Pose | Quality | Confidence |
|-------|---------------|---------|------------|
| `2.mp4` | Tadasana | Good | 96.8% |
| `1 (1).mp4` | Vrikshasana | Good | 100.0% |
| `20260217_165415.mp4` | Balasana | Good | 100.0% |

---

## Features

| Feature | Detail |
|---------|--------|
| **Pose Detection** | MediaPipe Pose — 33 body landmarks |
| **Feature Engineering** | 71 features: joint angles, pairwise distances, normalised coordinates |
| **Classifier** | Random Forest (300 estimators, GridSearchCV-tuned) |
| **Poses Supported** | 7 yoga poses |
| **Quality Levels** | Good · Average · Poor |
| **Total Classes** | 21 (7 × 3) |
| **Training Data** | 2,652 frames from 472 videos |
| **Inference** | Video files & static images |
| **Feedback** | Pose-specific correction tips |

---

## Dataset

472 videos across 7 poses, each annotated at 3 quality levels.

### Supported Poses

| Pose | Sanskrit Name |
|------|--------------|
| Child's Pose | Balasana |
| Cobra Pose | Bhujangasana |
| Lotus Pose | Padmasana |
| Mountain Pose (seated) | Parvatasana |
| Mountain Pose (standing) | Tadasana |
| Triangle Pose | Trikonasana |
| Tree Pose | Vrikshasana |

### Directory Structure

```
Final_project3_dataset/
├── balasana/
│   ├── good/
│   ├── avg/
│   └── poor/
├── bhujangasana/
│   ├── good/
│   ├── avg/
│   └── poor/
└── ...  (same for remaining 5 poses)
```

> **Note:** The dataset is not included in this repository due to size. Place your own videos following the structure above before training.

---

## Model Architecture

```
Video Input
    │
    ▼
Frame Extraction (1 fps via OpenCV)
    │
    ▼
MediaPipe Pose  →  33 Keypoints (x, y, z, visibility)
    │
    ▼
Feature Engineering
    ├── Joint Angles  (10 angles: elbows, knees, hips, shoulders)
    ├── Pairwise Distances  (12 distances between key body points)
    └── Normalised Coordinates  (33 × 3 = 99 raw values)
                                Total: 71 features
    │
    ▼
Random Forest Classifier
    ├── 300 estimators
    ├── Hyperparameters tuned via GridSearchCV
    └── 21-class output  (pose_quality, e.g. "vrikshasana_good")
    │
    ▼
Prediction + Correction Tip
```

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~95% |
| Training Samples | 2,652 frames |
| Features | 71 |
| Classes | 21 |
| Best Confidence (sample) | 100% (Vrikshasana, Balasana) |

Detailed per-class precision, recall, F1, and the confusion matrix are generated inside `training.ipynb` after execution.

---

## Project Structure

```
yoga-pose-classification/
│
├── training.ipynb          # End-to-end training pipeline (notebook)
├── predict.py              # CLI script — run predictions on new videos/images
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── LICENSE                 # MIT License
├── .gitignore
│
│   ── Generated after training ──
├── yoga_pose_model.pkl     # Trained Random Forest model
├── label_encoder.pkl       # Label encoder (21 classes)
├── features.csv            # Extracted feature dataset (2,652 rows)
└── frames/                 # Extracted video frames
```

> `yoga_pose_model.pkl`, `label_encoder.pkl`, `features.csv`, and `frames/` are excluded from the repo via `.gitignore` because of their size. Run the notebook to regenerate them.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/yoga-pose-classification.git
cd yoga-pose-classification

# 2. Create and activate a virtual environment
python -m venv mp_env

# Windows
mp_env\Scripts\activate

# Linux / macOS
source mp_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1 — Train the Model

Place your video dataset in `Final_project3_dataset/` (see [Dataset](#dataset) for folder structure), then:

```bash
jupyter notebook training.ipynb
```

Run all cells top-to-bottom. The notebook will:
- Extract frames from videos
- Detect keypoints with MediaPipe
- Engineer features and save `features.csv`
- Train and evaluate the Random Forest model
- Save `yoga_pose_model.pkl` and `label_encoder.pkl`

### 2 — Predict on a New Video

```bash
python predict.py --input path/to/your_video.mp4
```

**Sample output:**
```
==================================================
  Video       : your_video.mp4
  Pose        : Tadasana
  Quality     : Good
  Confidence  : 96.8%
  Frames analysed : 315

  Tip: Perfect posture! Keep it up.
==================================================
```

### 3 — Predict on a Static Image

```bash
python predict.py --input path/to/your_image.jpg
```

### 4 — Load the Model in Your Own Code

```python
import joblib
import cv2
import mediapipe as mp
import numpy as np

model = joblib.load("yoga_pose_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Pass a feature vector (71 values) to get a prediction
# features = extract_features(mediapipe_landmarks)  # See predict.py
prediction = model.predict([features])
label = label_encoder.inverse_transform(prediction)[0]
print(label)  # e.g. "tadasana_good"
```

---

## Future Work

- [ ] Real-time webcam inference with on-screen overlay
- [ ] Web / mobile application (Flask / Streamlit front-end)
- [ ] Expand dataset to more yoga poses
- [ ] Experiment with deep learning (LSTM on keypoint sequences, GNN)
- [ ] Audio feedback for correction tips

---

## Acknowledgements

- [MediaPipe](https://google.github.io/mediapipe/) — pose landmark detection
- [scikit-learn](https://scikit-learn.org/) — Random Forest and model evaluation
- [OpenCV](https://opencv.org/) — video frame extraction

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

If you have questions or suggestions, feel free to open an [Issue](https://github.com/<your-username>/yoga-pose-classification/issues) on this repository.
# yoga-pose-classification
