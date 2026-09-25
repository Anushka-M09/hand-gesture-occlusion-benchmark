"""Live interactive webcam demonstration and real-time occlusion stress-tester."""

from typing import Optional
import os
import sys
import time
import cv2
import numpy as np

from src.feature_extractor import HandLandmarkExtractor
from src.occluder import apply_block_occlusion
from src.model import load_model_bundle, DEFAULT_MODEL_PATH

OCCLUSION_PRESETS = [0.0, 0.10, 0.20, 0.30, 0.45, 0.60]


def run_live_webcam_test(
    model_path: str = DEFAULT_MODEL_PATH,
    camera_id: int = 0,
) -> None:
    """Launch real-time webcam testing window with live gesture prediction and occlusion toggles."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model file not found at '{model_path}'.")
        print("Please train the model first by running: python main.py --train")
        return

    print("\n========================================================")
    print(" Loading gesture classifier model bundle...")
    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    classes = bundle["classes"]
    print(f" Loaded model successfully! Recognized classes: {classes}")
    print("========================================================\n")

    print(f"[INFO] Opening webcam device (index {camera_id})...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"[ERROR] Unable to access camera device {camera_id}.")
        print("Please verify that your webcam is connected and not in use by another application.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Hand Gesture Recognition Live Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1080, 680)

    extractor = HandLandmarkExtractor(static_image_mode=False)

    current_occ_idx = 0
    show_skeleton = True
    fps_time = time.time()
    fps = 0.0

    print("\n================ Real-Time Webcam Controls ================")
    print("  [0 - 5] : Set Occlusion Level (0%, 10%, 20%, 30%, 45%, 60%)")
    print("  [+] / [-]: Increase / Decrease synthetic occlusion")
    print("  [S]     : Toggle Landmark Skeleton Overlay")
    print("  [Q / ESC]: Exit Live Demo")
    print("===========================================================\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Could not read frame from webcam.")
                break

            # Flip horizontally for natural mirror interaction
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Calculate FPS
            now = time.time()
            dt = now - fps_time
            if dt > 0.5:
                fps = 1.0 / max(1e-4, dt)
                fps_time = now

            occ_pct = OCCLUSION_PRESETS[current_occ_idx]

            # Detect hand bounding box on clean frame first (to target occlusion accurately)
            hand_bbox = extractor.extract_hand_bbox(frame)

            # Apply live synthetic occlusion if active
            if occ_pct > 0.0:
                display_frame = apply_block_occlusion(
                    image=frame,
                    occlusion_pct=occ_pct,
                    occlude_roi="random_hand_patch",
                    hand_bbox=hand_bbox,
                    occlusion_type="black",
                )
            else:
                display_frame = frame.copy()

            # Extract normalized landmarks from current (potentially occluded) frame
            landmarks = extractor.extract_landmarks(display_frame, normalize=True)

            predicted_label = "NO HAND DETECTED"
            confidence = 0.0
            tracking_status = "SEARCHING"
            status_color = (0, 0, 255)  # Red

            if landmarks is not None and len(landmarks) == 63:
                # Hand tracked successfully
                tracking_status = "TRACKING LOCKED"
                status_color = (0, 255, 0)  # Green

                # Predict gesture
                features = landmarks.reshape(1, -1)
                pred_code = model.predict(features)[0]
                predicted_label = label_encoder.inverse_transform([pred_code])[0].upper()

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)[0]
                    confidence = float(np.max(probs)) * 100.0
                else:
                    confidence = 100.0

                # Draw skeleton landmarks if enabled
                if show_skeleton:
                    display_frame, _ = extractor.draw_landmarks_on_image(display_frame)
            else:
                if occ_pct > 0.0 and hand_bbox is not None:
                    tracking_status = "DETECTION LOST (OCCLUDED)"
                    status_color = (0, 140, 255)  # Orange

            # ----------------- Sleek HUD Overlay -----------------
            # Top Status Bar
            cv2.rectangle(display_frame, (0, 0), (w, 110), (18, 18, 18), -1)
            cv2.line(display_frame, (0, 110), (w, 110), (0, 180, 255), 2)

            # Gesture Prediction Display
            cv2.putText(
                display_frame,
                f"GESTURE: {predicted_label}",
                (20, 42),
                cv2.FONT_HERSHEY_DUPLEX,
                1.1,
                (0, 255, 255) if tracking_status == "TRACKING LOCKED" else (160, 160, 160),
                2,
                cv2.LINE_AA,
            )

            # Confidence Bar
            if confidence > 0.0:
                bar_x, bar_y = 20, 60
                bar_w, bar_h = 240, 14
                cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                filled_w = int((confidence / 100.0) * bar_w)
                cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 220, 100), -1)
                cv2.putText(
                    display_frame,
                    f"Conf: {confidence:.1f}%",
                    (bar_x + bar_w + 12, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

            # Tracking Status & FPS (Top Right)
            cv2.putText(
                display_frame,
                f"STATUS: {tracking_status}",
                (w - 380, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                status_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"Occlusion: {int(occ_pct * 100)}% [Keys 0-5] | FPS: {fps:.1f}",
                (w - 380, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Bottom Controls Guide Bar
            cv2.rectangle(display_frame, (0, h - 38), (w, h), (18, 18, 18), -1)
            cv2.putText(
                display_frame,
                "Keys: [0-5] Occlusion (0%-60%) | [+/-] Step Occlusion | [S] Toggle Skeleton | [Q] Quit",
                (20, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Hand Gesture Recognition Live Test", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:  # Q or ESC
                break
            elif key in [ord(str(i)) for i in range(6)]:
                current_occ_idx = int(chr(key))
                print(f"[LIVE] Set Occlusion to: {int(OCCLUSION_PRESETS[current_occ_idx] * 100)}%")
            elif key in [ord("+"), ord("=")]:
                current_occ_idx = min(len(OCCLUSION_PRESETS) - 1, current_occ_idx + 1)
                print(f"[LIVE] Increased Occlusion to: {int(OCCLUSION_PRESETS[current_occ_idx] * 100)}%")
            elif key in [ord("-"), ord("_")]:
                current_occ_idx = max(0, current_occ_idx - 1)
                print(f"[LIVE] Decreased Occlusion to: {int(OCCLUSION_PRESETS[current_occ_idx] * 100)}%")
            elif key == ord("s"):
                show_skeleton = not show_skeleton

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        print("[INFO] Live webcam test closed.")


if __name__ == "__main__":
    run_live_webcam_test()
