"""MediaPipe Hands wrapper for 21-landmark extraction and invariant feature normalization.

Supports both modern MediaPipe Tasks API (Python 3.12/3.13+) and legacy mp.solutions.hands.
"""

from typing import Optional, Tuple, List
import os
import urllib.request
import numpy as np
import cv2
import mediapipe as mp

MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_TASK_PATH = os.path.join(os.path.dirname(__file__), "..", "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm Base
]


class HandLandmarkExtractor:
    """Extracts and normalizes 3D hand landmarks from images using MediaPipe.

    Includes dual-backend compatibility:
      1. Modern MediaPipe Tasks API (HandLandmarker)
      2. Legacy MediaPipe Solutions API (mp.solutions.hands)
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        model_task_path: Optional[str] = None,
    ) -> None:
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.backend = "unknown"

        # Check if legacy solutions API is available
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.backend = "solutions"
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
            )
        else:
            # Modern MediaPipe Tasks API
            self.backend = "tasks"
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python import BaseOptions

            task_path = model_task_path or DEFAULT_MODEL_TASK_PATH
            if not os.path.exists(task_path):
                print(f"[INFO] Downloading MediaPipe hand landmark model to '{task_path}'...")
                os.makedirs(os.path.dirname(os.path.abspath(task_path)), exist_ok=True)
                urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, task_path)
                print("[INFO] Model download complete.")

            base_options = BaseOptions(model_asset_path=task_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)

    def extract_landmarks(
        self,
        image: np.ndarray,
        normalize: bool = True,
    ) -> Optional[np.ndarray]:
        """Extract 21 3D landmarks from a BGR image and normalize coordinates.

        Mathematical Normalization Pipeline (for translation and scale invariance):
        1. Translation Invariance: Subtract wrist (Landmark 0) coordinates from all 21 points
           so Landmark 0 is always at (0, 0, 0).
        2. Scale Invariance: Divide by the maximum Euclidean distance from the wrist to any landmark,
           strictly bounding all normalized coordinates to [-1.0, 1.0].
        3. Flattening: Returns a 1D NumPy array of shape (63,).

        Returns:
            1D numpy array of shape (63,), or None if detector fails.
        """
        raw_coords = self._get_raw_landmarks(image)
        if raw_coords is None:
            return None

        if not normalize:
            return raw_coords.flatten()

        # Step 1: Translation Invariance - center at wrist
        wrist = raw_coords[0].copy()
        centered_coords = raw_coords - wrist

        # Step 2: Scale Invariance - divide by maximum distance from wrist
        distances = np.linalg.norm(centered_coords, axis=1)
        max_dist = np.max(distances)

        if max_dist > 1e-6:
            normalized_coords = centered_coords / max_dist
        else:
            normalized_coords = centered_coords

        # Step 3: Flatten to 1D vector (63,)
        return normalized_coords.flatten().astype(np.float32)

    def _get_raw_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Internal helper to extract raw (21, 3) landmarks across backends."""
        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.backend == "solutions":
            results = self.hands.process(rgb)
            if not results.multi_hand_landmarks:
                return None
            hand_lms = results.multi_hand_landmarks[0]
            raw = np.zeros((21, 3), dtype=np.float32)
            for i, lm in enumerate(hand_lms.landmark):
                raw[i] = [lm.x, lm.y, lm.z]
            return raw
        else:
            # Tasks API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = self.landmarker.detect(mp_image)
            if not results.hand_landmarks:
                return None
            hand_lms = results.hand_landmarks[0]
            raw = np.zeros((21, 3), dtype=np.float32)
            for i, lm in enumerate(hand_lms):
                raw[i] = [lm.x, lm.y, lm.z]
            return raw

    def extract_hand_bbox(
        self,
        image: np.ndarray,
        margin_pct: float = 0.05,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Extract pixel bounding box (xmin, ymin, xmax, ymax) of the hand."""
        raw = self._get_raw_landmarks(image)
        if raw is None:
            return None

        h, w = image.shape[:2]
        x_coords = raw[:, 0] * w
        y_coords = raw[:, 1] * h

        xmin, xmax = float(np.min(x_coords)), float(np.max(x_coords))
        ymin, ymax = float(np.min(y_coords)), float(np.max(y_coords))

        box_w = xmax - xmin
        box_h = ymax - ymin
        xmin = max(0, int(xmin - margin_pct * box_w))
        ymin = max(0, int(ymin - margin_pct * box_h))
        xmax = min(w - 1, int(xmax + margin_pct * box_w))
        ymax = min(h - 1, int(ymax + margin_pct * box_h))

        return xmin, ymin, xmax, ymax

    def draw_landmarks_on_image(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """Draw hand skeleton and keypoints cleanly on an OpenCV BGR image."""
        if image is None:
            return image, False

        raw = self._get_raw_landmarks(image)
        if raw is None:
            return image.copy(), False

        annotated = image.copy()
        h, w = image.shape[:2]
        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in raw]

        # Draw bone connections
        for idx1, idx2 in HAND_CONNECTIONS:
            cv2.line(annotated, pts[idx1], pts[idx2], (0, 220, 255), 2, cv2.LINE_AA)

        # Draw keypoints
        for i, (px, py) in enumerate(pts):
            # Wrist is green, fingertips are orange, intermediate joints are white
            if i == 0:
                color = (0, 255, 0)
                radius = 6
            elif i in [4, 8, 12, 16, 20]:
                color = (0, 140, 255)
                radius = 5
            else:
                color = (255, 255, 255)
                radius = 4
            cv2.circle(annotated, (px, py), radius, color, -1, cv2.LINE_AA)
            cv2.circle(annotated, (px, py), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

        return annotated, True

    def close(self) -> None:
        """Release detector resources."""
        if self.backend == "solutions" and hasattr(self, "hands"):
            self.hands.close()
        elif self.backend == "tasks" and hasattr(self, "landmarker"):
            self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
