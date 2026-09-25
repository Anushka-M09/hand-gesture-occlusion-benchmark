"""Interactive OpenCV webcam utility for clean gesture dataset collection and synthetic mock generator."""

from typing import List, Optional
import os
import sys
import time
import argparse
import numpy as np
import cv2

from src.feature_extractor import HandLandmarkExtractor

DEFAULT_GESTURES: List[str] = [
    "thumbs_up",
    "thumbs_down",
    "peace",
    "ok",
    "pointing",
    "fist",
]


class DatasetCollector:
    """Webcam collector for recording clean hand gesture samples with interactive UI."""

    def __init__(
        self,
        output_dir: str = "data/raw",
        gestures: Optional[List[str]] = None,
        samples_per_class: int = 50,
        camera_id: int = 0,
    ) -> None:
        self.output_dir = output_dir
        self.gestures = gestures or DEFAULT_GESTURES
        self.samples_per_class = samples_per_class
        self.camera_id = camera_id
        self.extractor = HandLandmarkExtractor(static_image_mode=False)

        for g in self.gestures:
            os.makedirs(os.path.join(self.output_dir, g), exist_ok=True)

    def run(self) -> None:
        """Run interactive collection loop."""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {self.camera_id}.")
            print("Tip: If you do not have a webcam available, use '--mock-data' to generate synthetic samples.")
            return

        current_class_idx = 0
        countdown_active = False
        countdown_start_time = 0.0
        burst_active = False
        captured_in_burst = 0
        burst_total = 20

        print("\n================ Webcam Gesture Collector ================")
        print("Controls:")
        print("  [SPACE]  : Start 3-second countdown to record burst")
        print("  [N]      : Switch to Next gesture class")
        print("  [P]      : Switch to Previous gesture class")
        print("  [Q / ESC]: Exit collector")
        print("==========================================================\n")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("[WARNING] Empty frame received.")
                    break

                # Flip horizontally for natural mirror feel
                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()
                h, w = frame.shape[:2]

                current_gesture = self.gestures[current_class_idx]
                class_dir = os.path.join(self.output_dir, current_gesture)
                existing_count = len([
                    f for f in os.listdir(class_dir)
                    if f.lower().endswith((".jpg", ".png"))
                ])

                # Live landmark visualization
                annotated, hand_detected = self.extractor.draw_landmarks_on_image(display_frame)
                if hand_detected:
                    display_frame = annotated

                # Draw UI Header Panel
                cv2.rectangle(display_frame, (0, 0), (w, 85), (20, 20, 20), -1)
                cv2.line(display_frame, (0, 85), (w, 85), (0, 200, 255), 2)

                # Class information
                cv2.putText(
                    display_frame,
                    f"Target: [{current_gesture.upper()}] ({current_class_idx + 1}/{len(self.gestures)})",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    f"Saved: {existing_count}/{self.samples_per_class} | Hand Detected: {'YES' if hand_detected else 'NO'}",
                    (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0) if hand_detected else (0, 0, 255),
                    2,
                )

                # Countdown logic
                now = time.time()
                if countdown_active:
                    elapsed = now - countdown_start_time
                    remaining = 3.0 - elapsed
                    if remaining > 0:
                        cv2.putText(
                            display_frame,
                            f"Recording in: {int(np.ceil(remaining))}",
                            (w // 2 - 140, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.6,
                            (0, 140, 255),
                            4,
                        )
                    else:
                        countdown_active = False
                        burst_active = True
                        captured_in_burst = 0

                # Burst recording logic
                if burst_active:
                    # Save frame
                    img_filename = f"{current_gesture}_{int(time.time() * 1000)}.jpg"
                    save_path = os.path.join(class_dir, img_filename)
                    cv2.imwrite(save_path, frame)
                    captured_in_burst += 1

                    # Visual flash indicator
                    cv2.rectangle(display_frame, (0, 0), (w, h), (0, 255, 0), 6)
                    cv2.putText(
                        display_frame,
                        f"RECORDING: {captured_in_burst}/{burst_total}",
                        (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        3,
                    )

                    if captured_in_burst >= burst_total:
                        burst_active = False
                        print(f"[INFO] Recorded {burst_total} samples for '{current_gesture}'.")

                # Bottom control guide
                cv2.rectangle(display_frame, (0, h - 35), (w, h), (15, 15, 15), -1)
                cv2.putText(
                    display_frame,
                    "SPACE: Capture Burst | N: Next Gesture | P: Prev Gesture | Q: Quit",
                    (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )

                cv2.imshow("Hand Gesture Dataset Collector", display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in [ord("q"), 27]:  # Q or ESC
                    break
                elif key == ord(" "):
                    if not countdown_active and not burst_active:
                        countdown_active = True
                        countdown_start_time = time.time()
                elif key == ord("n"):
                    current_class_idx = (current_class_idx + 1) % len(self.gestures)
                elif key == ord("p"):
                    current_class_idx = (current_class_idx - 1) % len(self.gestures)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.extractor.close()
            print("[INFO] Webcam collector closed.")


def generate_mock_dataset(
    output_dir: str = "data/raw",
    gestures: Optional[List[str]] = None,
    samples_per_class: int = 50,
) -> None:
    """Generate high-quality synthetic gesture images for pipeline verification without a webcam.

    Renders canonical hand landmark structures and silhouettes for:
    thumbs_up, thumbs_down, peace, ok, pointing, fist.
    """
    gestures = gestures or DEFAULT_GESTURES
    print(f"\n[INFO] Generating synthetic mock dataset in '{output_dir}'...")

    for gesture in gestures:
        gesture_dir = os.path.join(output_dir, gesture)
        os.makedirs(gesture_dir, exist_ok=True)

        for i in range(samples_per_class):
            img = _render_synthetic_gesture(gesture, variation_seed=i)
            file_path = os.path.join(gesture_dir, f"mock_{gesture}_{i:04d}.jpg")
            cv2.imwrite(file_path, img)

        print(f"  - Generated {samples_per_class} mock samples for [{gesture}]")

    print("[SUCCESS] Mock dataset generation complete!\n")


def _render_synthetic_gesture(gesture: str, variation_seed: int = 0) -> np.ndarray:
    """Render a synthetic colored hand canvas containing geometry corresponding to the target gesture."""
    np.random.seed(variation_seed)
    w, h = 480, 480

    # Natural varying background
    bg_color = (
        np.random.randint(190, 230),
        np.random.randint(190, 230),
        np.random.randint(190, 230),
    )
    img = np.full((h, w, 3), bg_color, dtype=np.uint8)

    # Slight camera noise
    noise = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Hand skin tones (varying realistic tones)
    skin_b = np.random.randint(110, 160)
    skin_g = np.random.randint(140, 185)
    skin_r = np.random.randint(190, 240)
    skin_color = (skin_b, skin_g, skin_r)
    outline_color = (max(0, skin_b - 35), max(0, skin_g - 35), max(0, skin_r - 35))

    cx, cy = 240 + np.random.randint(-15, 15), 260 + np.random.randint(-15, 15)

    # Palm base ellipse
    cv2.ellipse(img, (cx, cy), (65, 80), 0, 0, 360, skin_color, -1)
    cv2.ellipse(img, (cx, cy), (65, 80), 0, 0, 360, outline_color, 2)

    # Arm wrist base
    arm_pts = np.array([
        [cx - 45, cy + 60],
        [cx + 45, cy + 60],
        [cx + 50, h - 1],
        [cx - 50, h - 1]
    ], dtype=np.int32)
    cv2.fillPoly(img, [arm_pts], skin_color)
    cv2.polylines(img, [arm_pts], False, outline_color, 2)

    def draw_finger(start_pt, end_pt, thickness=26):
        cv2.line(img, start_pt, end_pt, skin_color, thickness, cv2.LINE_AA)
        cv2.circle(img, end_pt, thickness // 2, skin_color, -1, cv2.LINE_AA)
        cv2.circle(img, end_pt, thickness // 2, outline_color, 2, cv2.LINE_AA)

    # Gesture specific morphology
    if gesture == "thumbs_up":
        # Thumb pointing up
        draw_finger((cx - 35, cy - 20), (cx - 45, cy - 140), thickness=28)
        # Folded fingers (curled over palm)
        for fx in [cx - 15, cx + 10, cx + 35, cx + 55]:
            cv2.circle(img, (fx, cy - 30), 20, skin_color, -1)
            cv2.circle(img, (fx, cy - 30), 20, outline_color, 2)

    elif gesture == "thumbs_down":
        # Thumb pointing down
        draw_finger((cx - 35, cy + 20), (cx - 45, cy + 140), thickness=28)
        # Folded fingers
        for fx in [cx - 15, cx + 10, cx + 35, cx + 55]:
            cv2.circle(img, (fx, cy - 30), 20, skin_color, -1)
            cv2.circle(img, (fx, cy - 30), 20, outline_color, 2)

    elif gesture == "peace":
        # Index and Middle fingers forming 'V'
        draw_finger((cx - 15, cy - 40), (cx - 50, cy - 170), thickness=24)
        draw_finger((cx + 15, cy - 40), (cx + 50, cy - 170), thickness=24)
        # Folded thumb and others
        cv2.circle(img, (cx - 35, cy), 22, skin_color, -1)
        cv2.circle(img, (cx + 40, cy - 20), 20, skin_color, -1)

    elif gesture == "pointing":
        # Only Index finger extended upwards
        draw_finger((cx - 10, cy - 40), (cx - 10, cy - 180), thickness=25)
        # Folded others
        for fx in [cx - 35, cx + 15, cx + 40, cx + 60]:
            cv2.circle(img, (fx, cy - 20), 20, skin_color, -1)
            cv2.circle(img, (fx, cy - 20), 20, outline_color, 2)

    elif gesture == "ok":
        # Thumb and Index form circle
        cv2.circle(img, (cx - 30, cy - 60), 26, skin_color, -1)
        cv2.circle(img, (cx - 30, cy - 60), 12, bg_color, -1)
        # Other three fingers extended up
        draw_finger((cx + 10, cy - 30), (cx + 15, cy - 160), thickness=20)
        draw_finger((cx + 35, cy - 30), (cx + 45, cy - 150), thickness=20)
        draw_finger((cx + 55, cy - 30), (cx + 70, cy - 135), thickness=19)

    elif gesture == "fist":
        # All fingers folded in compact fist
        for fx, fy in [(cx - 30, cy - 40), (cx - 5, cy - 45), (cx + 20, cy - 45), (cx + 45, cy - 40)]:
            cv2.circle(img, (fx, fy), 24, skin_color, -1)
            cv2.circle(img, (fx, fy), 24, outline_color, 2)
        # Thumb wrapped over fingers
        cv2.line(img, (cx - 45, cy + 10), (cx + 15, cy - 20), skin_color, 28)
        cv2.circle(img, (cx + 15, cy - 20), 14, skin_color, -1)

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand Gesture Dataset Collector")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Target directory")
    parser.add_argument("--samples", type=int, default=50, help="Samples per gesture")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--mock", action="store_true", help="Generate synthetic mock images without webcam")
    args = parser.parse_args()

    if args.mock:
        generate_mock_dataset(output_dir=args.output_dir, samples_per_class=args.samples)
    else:
        collector = DatasetCollector(
            output_dir=args.output_dir,
            samples_per_class=args.samples,
            camera_id=args.camera,
        )
        collector.run()
