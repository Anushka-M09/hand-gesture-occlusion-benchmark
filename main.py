"""CLI pipeline runner connecting all steps of the Hand Gesture Occlusion Benchmark."""

from typing import List
import os
import sys
import argparse

from src.dataset_collector import DatasetCollector, generate_mock_dataset
from src.data_fetcher import fetch_hagrid_subset
from src.model import train_gesture_classifier, DEFAULT_MODEL_PATH
from src.benchmark import run_benchmark, DEFAULT_OUTPUT_DIR
from src.visualize import create_all_visualizations
from src.live_demo import run_live_webcam_test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robustness Benchmarking of Hand Gesture Recognition Under Partial Occlusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Execution Examples:
  # 1. Fetch public diverse dataset from Hugging Face (HaGRID subset):
  python main.py --fetch --samples 100

  # 2. Or record custom webcam samples interactively:
  python main.py --collect --samples 50

  # 3. Or generate instant synthetic mock dataset without webcam/internet:
  python main.py --mock-data --samples 40

  # 4. Train classifier on clean 0% occlusion data:
  python main.py --train --model-type random_forest

  # 5. Run systematic occlusion stress-test (0%, 10%, 20%, 30%, 45%, 60%):
  python main.py --benchmark

  # 6. Generate publication-ready figures & heatmaps:
  python main.py --visualize

  # 7. Run full pipeline end-to-end:
  python main.py --all
        """,
    )

    parser.add_argument("--demo", "--test-cam", dest="demo", action="store_true", help="Launch live interactive webcam recognition and occlusion test")
    parser.add_argument("--fetch", action="store_true", help="Download diverse public dataset from Hugging Face (HaGRID)")
    parser.add_argument("--collect", action="store_true", help="Launch interactive webcam gesture collector")
    parser.add_argument("--mock-data", action="store_true", help="Generate synthetic mock images for immediate testing")
    parser.add_argument("--train", action="store_true", help="Train ML classifier on clean landmarks")
    parser.add_argument("--benchmark", action="store_true", help="Stress test model across varying occlusion percentages")
    parser.add_argument("--visualize", action="store_true", help="Generate benchmark charts and confusion heatmaps")
    parser.add_argument("--all", action="store_true", help="Execute complete pipeline (fetch/mock -> train -> benchmark -> visualize)")

    # Configuration Options
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to clean raw gesture images")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for benchmark results and charts")
    parser.add_argument("--model-type", type=str, default="random_forest", choices=["random_forest", "mlp", "svm"], help="ML algorithm")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path for serialized model")
    parser.add_argument("--samples", type=int, default=50, help="Target samples per gesture class")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index for collection/demo")

    args = parser.parse_args()

    # If no flags passed, display help
    if not (args.demo or args.fetch or args.collect or args.mock_data or args.train or args.benchmark or args.visualize or args.all):
        parser.print_help()
        sys.exit(0)

    # Ensure required base directory structure exists
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs("data/occluded", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    print("\n====================================================================")
    print(" Robustness Benchmarking of Hand Gesture Recognition Under Occlusion")
    print("====================================================================\n")

    # Live Webcam Demo Mode
    if args.demo:
        print("[STEP] Launching real-time interactive webcam tester...")
        run_live_webcam_test(
            model_path=args.model_path,
            camera_id=args.camera,
        )
        return

    # Step 1: Data Gathering (Fetch, Collect, or Mock)
    if args.fetch:
        print("[STEP] Downloading diverse public dataset from Hugging Face...")
        fetch_hagrid_subset(
            output_dir=args.data_dir,
            samples_per_class=args.samples,
        )

    if args.collect:
        print("[STEP] Launching interactive webcam collector...")
        collector = DatasetCollector(
            output_dir=args.data_dir,
            samples_per_class=args.samples,
            camera_id=args.camera,
        )
        collector.run()

    if args.mock_data:
        print("[STEP] Generating synthetic mock dataset for rapid testing...")
        generate_mock_dataset(
            output_dir=args.data_dir,
            samples_per_class=args.samples,
        )

    # Step 2: Model Training
    if args.train or args.all:
        # Check if data exists; if running --all and no data exists, generate mock data or prompt
        classes_present = [
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d)) and not d.startswith(".")
        ]
        if not classes_present and args.all:
            print("[INFO] No dataset found in data/raw. Generating synthetic mock dataset to run --all...")
            generate_mock_dataset(output_dir=args.data_dir, samples_per_class=args.samples)

        print(f"\n[STEP] Training {args.model_type} classifier on clean hand landmarks...")
        train_gesture_classifier(
            data_dir=args.data_dir,
            model_type=args.model_type,
            model_save_path=args.model_path,
        )

    # Step 3: Occlusion Benchmarking
    if args.benchmark or args.all:
        print("\n[STEP] Running occlusion stress-test across thresholds [0%, 10%, 20%, 30%, 45%, 60%]...")
        run_benchmark(
            model_path=args.model_path,
            output_dir=args.output_dir,
        )

    # Step 4: Visualization
    if args.visualize or args.all:
        print("\n[STEP] Generating visualizations and performance curves...")
        csv_file = os.path.join(args.output_dir, "benchmark_results.csv")
        create_all_visualizations(
            csv_path=csv_file,
            output_dir=args.output_dir,
        )

    print("\n====================================================================")
    print(" Pipeline execution finished successfully!")
    print(f" Outputs and charts available in: {os.path.abspath(args.output_dir)}")
    print("====================================================================\n")


if __name__ == "__main__":
    main()
