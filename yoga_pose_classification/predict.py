"""
predict.py — Run yoga pose prediction on a new video or image.

Usage:
    python predict.py --input path/to/video.mp4
    python predict.py --input path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = Path("yoga_pose_model.pkl")
ENCODER_PATH = Path("label_encoder.pkl")

POSE_TIPS = {
    "tadasana": {
        "good": "Perfect posture! Keep it up.",
        "avg": "Try to straighten your spine more.",
        "poor": "Focus on aligning feet, hips, and shoulders.",
    },
    "vrikshasana": {
        "good": "Excellent balance!",
        "avg": "Try to raise your arms higher.",
        "poor": "Focus on balancing on one foot first.",
    },
    "balasana": {
        "good": "Great relaxation pose!",
        "avg": "Try to lower your hips closer to heels.",
        "poor": "Relax your shoulders and extend arms further.",
    },
    "bhujangasana": {
        "good": "Excellent cobra pose!",
        "avg": "Try to lift your chest higher.",
        "poor": "Keep your elbows close to your body.",
    },
    "padmasana": {
        "good": "Perfect lotus position!",
        "avg": "Try to sit more upright.",
        "poor": "Work on hip flexibility gradually.",
    },
    "parvatasana": {
        "good": "Perfect mountain pose!",
        "avg": "Try to straighten your arms fully.",
        "poor": "Focus on forming a straight line from hands to hips.",
    },
    "trikonasana": {
        "good": "Excellent triangle pose!",
        "avg": "Try to extend your arms in a straight line.",
        "poor": "Focus on keeping legs straight and hip-width apart.",
    },
}


# ── MediaPipe helpers ──────────────────────────────────────────────────────────
def _angle(a, b, c):
    """Angle in degrees at point b formed by a–b–c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def extract_features(landmarks):
    """Return a 1-D feature vector from MediaPipe pose landmarks."""
    lm = landmarks.landmark
    coords = [[l.x, l.y, l.z] for l in lm]

    # Raw coordinates (33 × 3)
    features = [v for pt in coords for v in pt]

    # Key joint indices (MediaPipe numbering)
    IDX = {
        "nose": 0, "l_shoulder": 11, "r_shoulder": 12,
        "l_elbow": 13,  "r_elbow": 14,
        "l_wrist": 15,  "r_wrist": 16,
        "l_hip": 23,    "r_hip": 24,
        "l_knee": 25,   "r_knee": 26,
        "l_ankle": 27,  "r_ankle": 28,
    }
    C = {k: coords[v] for k, v in IDX.items()}

    # Joint angles
    features += [
        _angle(C["l_shoulder"], C["l_elbow"], C["l_wrist"]),
        _angle(C["r_shoulder"], C["r_elbow"], C["r_wrist"]),
        _angle(C["l_elbow"], C["l_shoulder"], C["l_hip"]),
        _angle(C["r_elbow"], C["r_shoulder"], C["r_hip"]),
        _angle(C["l_shoulder"], C["l_hip"], C["l_knee"]),
        _angle(C["r_shoulder"], C["r_hip"], C["r_knee"]),
        _angle(C["l_hip"], C["l_knee"], C["l_ankle"]),
        _angle(C["r_hip"], C["r_knee"], C["r_ankle"]),
        _angle(C["l_shoulder"], C["nose"], C["r_shoulder"]),
        _angle(C["l_hip"], C["nose"], C["r_hip"]),
    ]

    # Pairwise distances
    pairs = [
        ("l_wrist", "r_wrist"), ("l_ankle", "r_ankle"),
        ("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"),
        ("nose", "l_hip"), ("nose", "r_hip"),
        ("l_wrist", "l_hip"), ("r_wrist", "r_hip"),
        ("l_ankle", "l_hip"), ("r_ankle", "r_hip"),
    ]
    for a, b in pairs:
        features.append(np.linalg.norm(np.array(C[a]) - np.array(C[b])))

    # Midpoints
    mid_shoulder = np.mean([C["l_shoulder"], C["r_shoulder"]], axis=0)
    mid_hip = np.mean([C["l_hip"], C["r_hip"]], axis=0)
    features.append(np.linalg.norm(mid_shoulder - mid_hip))
    features.append(np.linalg.norm(np.array(C["nose"]) - mid_hip))

    return features


# ── Core prediction logic ──────────────────────────────────────────────────────
def predict_image(frame, model, label_encoder, pose_detector):
    """Return (label, confidence) for a single BGR frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_detector.process(rgb)
    if not result.pose_landmarks:
        return None, 0.0
    feats = extract_features(result.pose_landmarks)
    proba = model.predict_proba([feats])[0]
    idx = np.argmax(proba)
    label = label_encoder.inverse_transform([idx])[0]
    return label, round(float(proba[idx]) * 100, 1)


def predict_video(video_path: Path, model, label_encoder):
    """Aggregate frame predictions and return majority vote result."""
    mp_pose = mp.solutions.pose
    counts: dict[str, int] = {}
    confidences: dict[str, list] = {}
    total_frames = 0

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        cap = cv2.VideoCapture(str(video_path))
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            label, conf = predict_image(frame, model, label_encoder, pose)
            if label:
                counts[label] = counts.get(label, 0) + 1
                confidences.setdefault(label, []).append(conf)
                total_frames += 1
        cap.release()

    if not counts:
        return None, 0.0, 0

    best = max(counts, key=counts.get)
    avg_conf = round(sum(confidences[best]) / len(confidences[best]), 1)
    return best, avg_conf, total_frames


def predict_single_image(image_path: Path, model, label_encoder):
    """Predict on a single image file."""
    mp_pose = mp.solutions.pose
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        label, conf = predict_image(frame, model, label_encoder, pose)
    return label, conf


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Yoga Pose Prediction")
    parser.add_argument("--input", required=True, help="Path to video (.mp4/.avi) or image (.jpg/.png)")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to model .pkl")
    parser.add_argument("--encoder", default=str(ENCODER_PATH), help="Path to label encoder .pkl")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        sys.exit(1)

    print("\nLoading model...")
    model = joblib.load(args.model)
    label_encoder = joblib.load(args.encoder)
    print("Model loaded successfully.\n")

    suffix = input_path.suffix.lower()
    if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
        label, conf, frames = predict_video(input_path, model, label_encoder)
        src_type = "Video"
        extra = f"  Frames analysed : {frames}"
    elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        label, conf = predict_single_image(input_path, model, label_encoder)
        src_type = "Image"
        extra = ""
    else:
        print(f"[ERROR] Unsupported file type: {suffix}")
        sys.exit(1)

    print("=" * 50)
    print(f"  {src_type}     : {input_path.name}")
    if label:
        pose, quality = label.rsplit("_", 1)
        tip = POSE_TIPS.get(pose, {}).get(quality, "")
        print(f"  Pose        : {pose.capitalize()}")
        print(f"  Quality     : {quality.capitalize()}")
        print(f"  Confidence  : {conf}%")
        if extra:
            print(extra)
        if tip:
            print(f"\n  Tip: {tip}")
    else:
        print("  No pose detected — ensure the subject is clearly visible.")
    print("=" * 50)


if __name__ == "__main__":
    main()
