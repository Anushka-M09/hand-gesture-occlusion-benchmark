"""Gesture classification model training and evaluation using Scikit-Learn."""

from typing import Dict, List, Optional, Tuple, Any
import os
import json
import time
import argparse
import joblib
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from src.feature_extractor import HandLandmarkExtractor

DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "gesture_classifier.joblib")
DEFAULT_SPLIT_PATH = os.path.join(DEFAULT_MODEL_DIR, "test_split.json")


def load_dataset_features(
    data_dir: str = "data/raw",
    valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Scan raw gesture directories and extract normalized 63-d landmark vectors.

    Args:
        data_dir: Root directory with subfolders for each gesture class.
        valid_extensions: Allowed image file extensions.

    Returns:
        X: numpy array of shape (N, 63)
        y: numpy array of string labels of shape (N,)
        valid_paths: list of file paths successfully processed
        classes: sorted list of class names found
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    ])

    if not classes:
        raise ValueError(f"No gesture class subdirectories found in '{data_dir}'.")

    print(f"\n[INFO] Found {len(classes)} gesture classes: {classes}")
    extractor = HandLandmarkExtractor(static_image_mode=True)

    features_list: List[np.ndarray] = []
    labels_list: List[str] = []
    valid_paths: List[str] = []

    total_images = 0
    undetected_count = 0

    try:
        for class_name in classes:
            class_folder = os.path.join(data_dir, class_name)
            files = [
                f for f in os.listdir(class_folder)
                if f.lower().endswith(valid_extensions)
            ]
            print(f"  Extracting landmarks for '{class_name}' ({len(files)} files)...")

            for fname in files:
                total_images += 1
                img_path = os.path.join(class_folder, fname)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                landmarks = extractor.extract_landmarks(img, normalize=True)
                if landmarks is not None and len(landmarks) == 63:
                    features_list.append(landmarks)
                    labels_list.append(class_name)
                    valid_paths.append(img_path)
                else:
                    undetected_count += 1

    finally:
        extractor.close()

    print(f"\n[SUMMARY] Processed {total_images} total clean images.")
    print(f"  - Successfully extracted landmarks: {len(features_list)} ({len(features_list)/(total_images or 1)*100:.1f}%)")
    print(f"  - Landmark detection failed on: {undetected_count} clean frames.")

    if len(features_list) == 0:
        raise RuntimeError("No hand landmarks were detected in the dataset. Cannot train model.")

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list)
    return X, y, valid_paths, classes


def train_gesture_classifier(
    data_dir: str = "data/raw",
    model_type: str = "random_forest",
    model_save_path: str = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train machine learning classifier on clean 63-d normalized landmarks.

    Args:
        data_dir: Root directory of clean gesture images.
        model_type: Classifier type ('random_forest', 'mlp', 'svm').
        model_save_path: Destination path for serialized model bundle.
        test_size: Fraction of data held out for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing trained classifier, evaluation metrics, and test split paths.
    """
    X, y, paths, classes = load_dataset_features(data_dir=data_dir)

    # Encode string labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split with stratification
    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=test_size,
        stratify=y_encoded,
        random_state=random_state,
    )

    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y_encoded[idx_train], y_encoded[idx_test]
    test_paths = [paths[i] for i in idx_test]
    test_labels = [y[i] for i in idx_test]

    print(f"\n[INFO] Split dataset into {len(X_train)} training and {len(X_test)} testing samples.")

    # Initialize classifier
    if model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_split=2,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=400,
            random_state=random_state,
        )
    elif model_type == "svm":
        clf = SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f"[INFO] Training {model_type} classifier...")
    clf.fit(X_train, y_train)

    # Evaluate on clean test set
    y_pred = clf.predict(X_test)
    baseline_acc = accuracy_score(y_test, y_pred)
    target_names = label_encoder.classes_.tolist()

    print("\n================ Baseline Test Set Performance ================")
    print(f"Overall Baseline Accuracy (0% Occlusion): {baseline_acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    print("=================================================================\n")

    # Serialize model bundle
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    bundle = {
        "model": clf,
        "label_encoder": label_encoder,
        "classes": target_names,
        "model_type": model_type,
        "feature_dim": 63,
        "baseline_accuracy": float(baseline_acc),
        "trained_timestamp": time.time(),
    }
    joblib.dump(bundle, model_save_path)
    print(f"[SUCCESS] Trained model bundle saved to: {model_save_path}")

    # Save test split paths and labels for reproducible benchmarking
    split_info = {
        "test_paths": test_paths,
        "test_labels": test_labels,
        "classes": target_names,
    }
    with open(DEFAULT_SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    print(f"[INFO] Saved test split manifest to: {DEFAULT_SPLIT_PATH}")

    return {
        "model": clf,
        "label_encoder": label_encoder,
        "baseline_accuracy": baseline_acc,
        "test_paths": test_paths,
        "test_labels": test_labels,
    }


def load_model_bundle(model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """Load serialized model bundle from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please run training first.")
    return joblib.load(model_path)


def predict_gesture(
    image: np.ndarray,
    model_bundle: Optional[Dict[str, Any]] = None,
    model_path: str = DEFAULT_MODEL_PATH,
) -> Tuple[Optional[str], Optional[float]]:
    """Predict hand gesture from a single BGR image.

    Returns:
        (predicted_class_name, confidence) or (None, None) if detection fails.
    """
    if model_bundle is None:
        model_bundle = load_model_bundle(model_path)

    extractor = HandLandmarkExtractor(static_image_mode=True)
    landmarks = extractor.extract_landmarks(image, normalize=True)
    extractor.close()

    if landmarks is None:
        return None, None

    features = landmarks.reshape(1, -1)
    model = model_bundle["model"]
    encoder = model_bundle["label_encoder"]

    pred_idx = model.predict(features)[0]
    pred_label = encoder.inverse_transform([pred_idx])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        confidence = float(np.max(probs))

    return pred_label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hand Gesture Recognition Model")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to clean raw dataset")
    parser.add_argument("--model-type", type=str, default="random_forest", choices=["random_forest", "mlp", "svm"])
    parser.add_argument("--save-path", type=str, default=DEFAULT_MODEL_PATH, help="Model export destination")
    args = parser.parse_args()

    train_gesture_classifier(
        data_dir=args.data_dir,
        model_type=args.model_type,
        model_save_path=args.save_path,
    )
