"""Systematic robustness stress-testing across synthetic occlusion thresholds."""

from typing import Dict, List, Optional, Tuple, Any
import os
import json
import argparse
import numpy as np
import pandas as pd
import cv2

from src.feature_extractor import HandLandmarkExtractor
from src.occluder import apply_block_occlusion
from src.model import load_model_bundle, DEFAULT_MODEL_PATH, DEFAULT_SPLIT_PATH

DEFAULT_OCCLUSION_LEVELS: List[float] = [0.0, 0.10, 0.20, 0.30, 0.45, 0.60]
DEFAULT_OUTPUT_DIR = "outputs"


class OcclusionBenchmark:
    """Stress-test hand gesture classifier robustness under synthetic contiguous block occlusions."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        occlusion_levels: Optional[List[float]] = None,
        save_visual_samples: bool = True,
    ) -> None:
        self.model_path = model_path
        self.output_dir = output_dir
        self.occlusion_levels = occlusion_levels or DEFAULT_OCCLUSION_LEVELS
        self.save_visual_samples = save_visual_samples

        self.model_bundle = load_model_bundle(model_path)
        self.classifier = self.model_bundle["model"]
        self.label_encoder = self.model_bundle["label_encoder"]
        self.classes = self.model_bundle["classes"]

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("data/occluded", exist_ok=True)

    def load_test_dataset(
        self,
        test_split_path: str = DEFAULT_SPLIT_PATH,
        fallback_data_dir: str = "data/raw",
        max_samples_per_class: int = 40,
    ) -> List[Tuple[str, str]]:
        """Load test image paths and their corresponding ground truth labels."""
        if os.path.exists(test_split_path):
            with open(test_split_path, "r", encoding="utf-8") as f:
                split_info = json.load(f)
            paths = split_info.get("test_paths", [])
            labels = split_info.get("test_labels", [])
            valid_pairs = [(p, l) for p, l in zip(paths, labels) if os.path.exists(p)]
            if valid_pairs:
                print(f"[INFO] Loaded {len(valid_pairs)} test samples from '{test_split_path}'.")
                return valid_pairs

        print(f"[INFO] Manifest not found. Sampling up to {max_samples_per_class} images per class from '{fallback_data_dir}'...")
        pairs: List[Tuple[str, str]] = []
        if not os.path.exists(fallback_data_dir):
            raise FileNotFoundError(f"Fallback directory '{fallback_data_dir}' does not exist.")

        for cls_name in sorted(os.listdir(fallback_data_dir)):
            cls_dir = os.path.join(fallback_data_dir, cls_name)
            if not os.path.isdir(cls_dir) or cls_name.startswith("."):
                continue
            images = [
                os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            for p in images[:max_samples_per_class]:
                pairs.append((p, cls_name))

        print(f"[INFO] Loaded {len(pairs)} test samples from directory tree.")
        return pairs

    def run_benchmark(
        self,
        test_data: Optional[List[Tuple[str, str]]] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Execute systematic benchmark across all occlusion levels.

        Records for each frame:
            - image_path
            - ground_truth
            - occlusion_pct
            - detection_success (bool: True if MediaPipe extracted landmarks)
            - predicted (predicted label or "UNDETECTED")
            - is_correct (bool)
            - confidence (float or 0.0)

        Returns:
            Pandas DataFrame containing all trial results.
        """
        if test_data is None:
            test_data = self.load_test_dataset()

        if not test_data:
            raise ValueError("No test images available for benchmarking.")

        extractor = HandLandmarkExtractor(static_image_mode=True)
        results: List[Dict[str, Any]] = []

        print("\n================ Starting Occlusion Robustness Stress Test ================")
        print(f"Total Test Images: {len(test_data)}")
        print(f"Occlusion Levels : {[f'{int(p*100)}%' for p in self.occlusion_levels]}")
        print("===========================================================================\n")

        # Precompute or cache clean hand bounding boxes for targeted occlusion
        print("[INFO] Computing reference hand ROIs on clean test frames...")
        test_cache: List[Dict[str, Any]] = []
        for img_path, ground_truth in test_data:
            img = cv2.imread(img_path)
            if img is None:
                continue
            bbox = extractor.extract_hand_bbox(img)
            test_cache.append({
                "path": img_path,
                "label": ground_truth,
                "image": img,
                "bbox": bbox,
            })

        print(f"[INFO] Successfully cached {len(test_cache)} valid test frames.\n")

        saved_sample_counts: Dict[float, int] = {p: 0 for p in self.occlusion_levels}

        for level_idx, occ_pct in enumerate(self.occlusion_levels):
            occ_label = f"{int(occ_pct * 100)}%"
            print(f"--> Evaluating Occlusion Level: {occ_label} ({level_idx + 1}/{len(self.occlusion_levels)})...")

            level_detected = 0
            level_correct = 0

            for sample_idx, item in enumerate(test_cache):
                clean_img = item["image"]
                gt_label = item["label"]
                hand_bbox = item["bbox"]

                # Apply synthetic occlusion
                if occ_pct == 0.0:
                    test_img = clean_img.copy()
                else:
                    test_img = apply_block_occlusion(
                        image=clean_img,
                        occlusion_pct=occ_pct,
                        occlude_roi="random_hand_patch",
                        hand_bbox=hand_bbox,
                        occlusion_type="black",
                        seed=seed + sample_idx,
                    )

                # Save sample visual preview if enabled
                if self.save_visual_samples and saved_sample_counts[occ_pct] < 4:
                    out_filename = f"occ_{int(occ_pct * 100):02d}pct_{gt_label}_{saved_sample_counts[occ_pct]}.jpg"
                    out_filepath = os.path.join("data/occluded", out_filename)
                    cv2.imwrite(out_filepath, test_img)
                    saved_sample_counts[occ_pct] += 1

                # Attempt MediaPipe landmark extraction
                landmarks = extractor.extract_landmarks(test_img, normalize=True)

                if landmarks is None or len(landmarks) != 63:
                    # MediaPipe failure due to occlusion
                    detection_success = False
                    predicted_label = "UNDETECTED"
                    confidence = 0.0
                    is_correct = False
                else:
                    detection_success = True
                    level_detected += 1
                    features = landmarks.reshape(1, -1)
                    pred_code = self.classifier.predict(features)[0]
                    predicted_label = self.label_encoder.inverse_transform([pred_code])[0]

                    if hasattr(self.classifier, "predict_proba"):
                        probs = self.classifier.predict_proba(features)[0]
                        confidence = float(np.max(probs))
                    else:
                        confidence = 1.0

                    is_correct = bool(predicted_label == gt_label)
                    if is_correct:
                        level_correct += 1

                results.append({
                    "image_path": item["path"],
                    "ground_truth": gt_label,
                    "occlusion_pct": occ_pct,
                    "occlusion_str": occ_label,
                    "detection_success": detection_success,
                    "predicted": predicted_label,
                    "is_correct": is_correct,
                    "confidence": confidence,
                })

            total_level_samples = len(test_cache)
            det_rate = (level_detected / total_level_samples) * 100
            acc_rate = (level_correct / total_level_samples) * 100
            cond_acc = (level_correct / (level_detected or 1)) * 100
            print(f"    Detection Rate: {det_rate:.1f}% | Overall Acc: {acc_rate:.1f}% | Cond Acc (When Detected): {cond_acc:.1f}%")

        extractor.close()
        df = pd.DataFrame(results)

        # Save results to CSV
        csv_path = os.path.join(self.output_dir, "benchmark_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[SUCCESS] Benchmark complete! Results saved to '{csv_path}'.")

        self.print_summary_tables(df)
        return df

    def print_summary_tables(self, df: pd.DataFrame) -> None:
        """Display consolidated metrics per occlusion percentage."""
        print("\n================ Robustness Benchmark Summary ================")

        summary = []
        for occ_pct in self.occlusion_levels:
            sub = df[df["occlusion_pct"] == occ_pct]
            total = len(sub)
            detected = sub["detection_success"].sum()
            correct = sub["is_correct"].sum()

            det_rate = (detected / total) * 100 if total > 0 else 0.0
            overall_acc = (correct / total) * 100 if total > 0 else 0.0
            cond_acc = (correct / detected) * 100 if detected > 0 else 0.0

            summary.append({
                "Occlusion": f"{int(occ_pct * 100)}%",
                "Total Frames": total,
                "Detected": detected,
                "Detection Rate (%)": round(det_rate, 2),
                "Correct": correct,
                "Overall Acc (%)": round(overall_acc, 2),
                "Cond Acc (%)": round(cond_acc, 2),
            })

        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))

        summary_csv = os.path.join(self.output_dir, "benchmark_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"Summary table exported to: '{summary_csv}'")
        print("==============================================================\n")


def run_benchmark(
    model_path: str = DEFAULT_MODEL_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    occlusion_levels: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Convenience function to instantiate and run benchmark."""
    benchmarker = OcclusionBenchmark(
        model_path=model_path,
        output_dir=output_dir,
        occlusion_levels=occlusion_levels,
    )
    return benchmarker.run_benchmark()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Occlusion Benchmark Stress Test")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    run_benchmark(model_path=args.model_path, output_dir=args.output_dir)
