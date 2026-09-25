"""Visualization module for benchmark curves, failure analysis, and confusion matrices."""

from typing import List, Optional
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2

# Configure clean aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_accuracy_vs_occlusion(
    df: pd.DataFrame,
    output_path: str = "outputs/accuracy_vs_occlusion.png",
) -> None:
    """Plot overall and per-gesture accuracy degradation across occlusion levels."""
    plt.figure(figsize=(10, 6), dpi=300)

    # Sort occlusion percentages
    occlusions = sorted(df["occlusion_pct"].unique())
    occ_labels = [f"{int(p * 100)}%" for p in occlusions]

    # Calculate overall accuracy
    overall_acc = []
    for occ in occlusions:
        sub = df[df["occlusion_pct"] == occ]
        acc = (sub["is_correct"].sum() / len(sub)) * 100 if len(sub) > 0 else 0.0
        overall_acc.append(acc)

    # Plot bold overall accuracy line
    plt.plot(
        occ_labels,
        overall_acc,
        marker="o",
        linewidth=3.5,
        color="#111827",
        label="Overall Accuracy (All Classes)",
        zorder=5,
    )

    # Palette for individual gesture curves
    classes = sorted(df["ground_truth"].unique())
    palette = sns.color_palette("tab10", n_colors=len(classes))

    for idx, cls_name in enumerate(classes):
        cls_acc = []
        for occ in occlusions:
            sub = df[(df["occlusion_pct"] == occ) & (df["ground_truth"] == cls_name)]
            acc = (sub["is_correct"].sum() / len(sub)) * 100 if len(sub) > 0 else 0.0
            cls_acc.append(acc)

        plt.plot(
            occ_labels,
            cls_acc,
            marker="s",
            markersize=5,
            linewidth=1.8,
            linestyle="--",
            alpha=0.85,
            color=palette[idx],
            label=f"{cls_name}",
        )

    plt.title("Hand Gesture Recognition Accuracy vs. Partial Occlusion", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Synthetic Hand Occlusion Percentage", fontsize=12, labelpad=10)
    plt.ylabel("Classification Accuracy (%)", fontsize=12, labelpad=10)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Accuracy vs Occlusion curve -> {output_path}")


def plot_detection_dropoff(
    df: pd.DataFrame,
    output_path: str = "outputs/detection_dropoff.png",
) -> None:
    """Plot MediaPipe landmark detection failure curve vs occlusion percentage."""
    plt.figure(figsize=(9, 5.5), dpi=300)

    occlusions = sorted(df["occlusion_pct"].unique())
    occ_labels = [f"{int(p * 100)}%" for p in occlusions]

    detection_rates = []
    failure_rates = []

    for occ in occlusions:
        sub = df[df["occlusion_pct"] == occ]
        total = len(sub)
        detected = sub["detection_success"].sum()
        det_rate = (detected / total) * 100 if total > 0 else 0.0
        fail_rate = 100.0 - det_rate
        detection_rates.append(det_rate)
        failure_rates.append(fail_rate)

    # Plot Curves
    plt.plot(
        occ_labels,
        detection_rates,
        marker="o",
        linewidth=2.8,
        color="#059669",
        label="Detection Success Rate (%)",
    )
    plt.plot(
        occ_labels,
        failure_rates,
        marker="x",
        markersize=8,
        linewidth=2.5,
        color="#DC2626",
        linestyle="-.",
        label="Detection Failure / Landmark Lost (%)",
    )

    # Highlight critical degradation threshold (drop below 50%)
    for i, rate in enumerate(detection_rates):
        if rate < 50.0:
            plt.axvline(x=i, color="#9CA3AF", linestyle=":", linewidth=1.5)
            plt.annotate(
                f"Critical Threshold ({occ_labels[i]})\nDet: {rate:.1f}%",
                xy=(i, rate),
                xytext=(i - 0.4, rate + 15),
                arrowprops=dict(facecolor="#374151", shrink=0.08, width=1, headwidth=6),
                fontsize=9,
                fontweight="semibold",
                backgroundcolor="white",
            )
            break

    plt.title("MediaPipe Landmark Detector Drop-off Under Occlusion", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Hand Occlusion Percentage", fontsize=12, labelpad=10)
    plt.ylabel("Rate (%)", fontsize=12, labelpad=10)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10, loc="center right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Detection dropoff curve -> {output_path}")


def plot_confusion_matrices(
    df: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
    output_path: str = "outputs/confusion_matrices.png",
) -> None:
    """Plot side-by-side heatmaps of confusion matrices at key occlusion thresholds (e.g. 0%, 30%, 60%)."""
    thresholds = thresholds or [0.0, 0.30, 0.60]

    # Available occlusion percentages in dataset
    available_occs = sorted(df["occlusion_pct"].unique())
    # Match requested thresholds to closest available
    selected_occs = []
    for t in thresholds:
        closest = min(available_occs, key=lambda x: abs(x - t))
        if closest not in selected_occs:
            selected_occs.append(closest)

    classes = sorted(df["ground_truth"].unique())
    # Extended class labels including UNDETECTED
    eval_labels = classes + ["UNDETECTED"]

    fig, axes = plt.subplots(1, len(selected_occs), figsize=(6 * len(selected_occs), 5.5), dpi=300)
    if len(selected_occs) == 1:
        axes = [axes]

    for idx, (occ, ax) in enumerate(zip(selected_occs, axes)):
        sub = df[df["occlusion_pct"] == occ]
        y_true = sub["ground_truth"]
        y_pred = sub["predicted"]

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=eval_labels)
        # We only want ground truth rows for actual classes (not UNDETECTED)
        cm_display = cm[:len(classes), :]

        sns.heatmap(
            cm_display,
            annot=True,
            fmt="d",
            cmap="Blues" if idx == 0 else ("YlOrRd" if idx == len(selected_occs) - 1 else "Oranges"),
            xticklabels=eval_labels,
            yticklabels=classes,
            cbar=False,
            ax=ax,
            linewidths=0.5,
            linecolor="#e5e7eb",
        )

        acc = (sub["is_correct"].sum() / len(sub)) * 100 if len(sub) > 0 else 0.0
        det = (sub["detection_success"].sum() / len(sub)) * 100 if len(sub) > 0 else 0.0

        ax.set_title(
            f"Occlusion: {int(occ * 100)}%\nAccuracy: {acc:.1f}% | Detection: {det:.1f}%",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Predicted Class", fontsize=10, labelpad=8)
        if idx == 0:
            ax.set_ylabel("Ground Truth Class", fontsize=10, labelpad=8)
        else:
            ax.set_ylabel("")

        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Confusion Matrices Across Increasing Occlusion Levels", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Confusion matrices -> {output_path}")


def generate_visual_samples_grid(
    occluded_dir: str = "data/occluded",
    output_path: str = "outputs/occlusion_visual_grid.png",
) -> None:
    """Assemble a montage grid demonstrating the simulated occlusion levels."""
    if not os.path.exists(occluded_dir):
        return

    images = sorted([
        f for f in os.listdir(occluded_dir)
        if f.lower().endswith((".jpg", ".png"))
    ])

    if not images:
        return

    # Select up to 6 diverse samples
    samples = images[:min(6, len(images))]
    fig, axes = plt.subplots(1, len(samples), figsize=(3.5 * len(samples), 3.8), dpi=300)
    if len(samples) == 1:
        axes = [axes]

    for idx, (img_name, ax) in enumerate(zip(samples, axes)):
        img_path = os.path.join(occluded_dir, img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.axis("off")
        ax.set_title(img_name.replace(".jpg", "").replace("_", " "), fontsize=9)

    fig.suptitle("Visual Inspection of Synthetic Hand Occlusions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Occlusion visual grid -> {output_path}")


def create_all_visualizations(
    csv_path: str = "outputs/benchmark_results.csv",
    output_dir: str = "outputs",
) -> None:
    """Load benchmark results CSV and generate all required plots."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Benchmark results file '{csv_path}' not found. Run benchmark first.")

    df = pd.read_csv(csv_path)
    print(f"\n[INFO] Generating benchmark plots from '{csv_path}' ({len(df)} records)...")

    plot_accuracy_vs_occlusion(df, os.path.join(output_dir, "accuracy_vs_occlusion.png"))
    plot_detection_dropoff(df, os.path.join(output_dir, "detection_dropoff.png"))
    plot_confusion_matrices(df, thresholds=[0.0, 0.30, 0.60], output_path=os.path.join(output_dir, "confusion_matrices.png"))
    generate_visual_samples_grid(output_path=os.path.join(output_dir, "occlusion_visual_grid.png"))

    print("[SUCCESS] All visualization figures successfully generated!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Visualizations for Occlusion Benchmark")
    parser.add_argument("--csv", type=str, default="outputs/benchmark_results.csv")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    create_all_visualizations(csv_path=args.csv, output_dir=args.output_dir)
