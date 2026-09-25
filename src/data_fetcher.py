"""Automated high-throughput data fetching module to download diverse public hand gesture datasets from Hugging Face."""

from typing import Dict, List, Optional
import os
import sys
import argparse
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image

# Mapping from HaGRID dataset directory names to project target class names
HAGRID_CLASS_MAPPING: Dict[str, str] = {
    "like": "thumbs_up",
    "dislike": "thumbs_down",
    "peace": "peace",
    "ok": "ok",
    "one": "pointing",
    "point": "pointing",
    "fist": "fist",
}

DEFAULT_CLASSES: List[str] = [
    "thumbs_up",
    "thumbs_down",
    "peace",
    "ok",
    "pointing",
    "fist",
]


def _download_and_save_image(
    session: requests.Session,
    remote_file: str,
    dest_path: str,
    dataset_name: str,
    max_dimension: int = 640,
) -> bool:
    """Download a single image via HTTP, optimize dimensions, and save to disk."""
    url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{remote_file}"
    try:
        resp = session.get(url, allow_redirects=True, timeout=20)
        if resp.status_code != 200:
            return False

        img = Image.open(BytesIO(resp.content))
        img = img.convert("RGB")

        # Downsample large 1920x1080 images for fast training/inference while maintaining fidelity
        w, h = img.size
        if max(w, h) > max_dimension:
            scale = max_dimension / float(max(w, h))
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        img.save(dest_path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def fetch_hagrid_subset(
    output_dir: str = "data/raw",
    samples_per_class: int = 200,
    dataset_name: str = "s17660101713/hagrid-subset",
    classes: Optional[List[str]] = None,
    max_workers: int = 8,
) -> Dict[str, int]:
    """Download diverse hand gesture images directly from Hugging Face HaGRID subset.

    Uses high-throughput parallel workers to stream and standardize diverse hand gestures
    representing different subjects, skin tones, lighting conditions, and camera angles.

    Args:
        output_dir: Destination directory for raw images.
        samples_per_class: Number of clean images to save per target gesture class.
        dataset_name: Hugging Face dataset identifier (e.g. 's17660101713/hagrid-subset').
        classes: Target gesture class names. Defaults to DEFAULT_CLASSES.
        max_workers: Concurrent thread pool workers.

    Returns:
        Dictionary mapping class names to number of downloaded images.
    """
    target_classes = classes or DEFAULT_CLASSES
    print(f"\n========================================================")
    print(f" Fetching Diverse Public Dataset: {dataset_name}")
    print(f" Target Classes   : {target_classes}")
    print(f" Samples Per Class: {samples_per_class}")
    print(f" Output Directory : {output_dir}")
    print(f" Concurrent Threads: {max_workers}")
    print(f"========================================================\n")

    # Ensure output folders exist
    counts: Dict[str, int] = {c: 0 for c in target_classes}
    for c in target_classes:
        folder = os.path.join(output_dir, c)
        os.makedirs(folder, exist_ok=True)
        existing = len([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("mock_")
        ])
        counts[c] = existing

    if all(counts[c] >= samples_per_class for c in target_classes):
        print(f"[INFO] All classes already have >= {samples_per_class} images in '{output_dir}'.")
        return counts

    print("[INFO] Querying repository manifest from Hugging Face Hub API...")
    api_url = f"https://huggingface.co/api/datasets/{dataset_name}"

    try:
        resp = requests.get(api_url, timeout=20)
        resp.raise_for_status()
        siblings = resp.json().get("siblings", [])
        all_files = [s["rfilename"] for s in siblings if s.get("rfilename", "").lower().endswith((".jpg", ".png"))]
    except Exception as e:
        print(f"[ERROR] Failed to query Hugging Face API: {e}")
        print("[TIP] You can also run 'python main.py --mock-data' for offline testing.")
        return counts

    # Group files by mapped class
    remote_by_class: Dict[str, List[str]] = {c: [] for c in target_classes}
    for file_path in all_files:
        parts = file_path.split("/")
        if len(parts) >= 3:
            raw_category = parts[2].lower()
            mapped = HAGRID_CLASS_MAPPING.get(raw_category)
            if mapped and mapped in target_classes:
                remote_by_class[mapped].append(file_path)

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers * 2)
    session.mount("https://", adapter)

    for cls_name in target_classes:
        needed = samples_per_class - counts[cls_name]
        if needed <= 0:
            print(f"  - [{cls_name}]: Already complete ({counts[cls_name]} images).")
            continue

        candidates = remote_by_class.get(cls_name, [])
        target_dir = os.path.join(output_dir, cls_name)
        downloaded = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_dest = {}
            # Submit enough candidates to reach target
            candidates_to_submit = candidates[:needed * 2]
            for idx, remote_file in enumerate(candidates_to_submit):
                dest_filename = f"hagrid_{cls_name}_{counts[cls_name] + idx + 1:04d}.jpg"
                dest_path = os.path.join(target_dir, dest_filename)
                fut = executor.submit(
                    _download_and_save_image,
                    session,
                    remote_file,
                    dest_path,
                    dataset_name,
                )
                future_to_dest[fut] = dest_path

            for fut in as_completed(future_to_dest):
                if counts[cls_name] >= samples_per_class:
                    continue
                if fut.result():
                    counts[cls_name] += 1
                    downloaded += 1
                    if downloaded % 10 == 0 or counts[cls_name] >= samples_per_class:
                        print(f"    [{cls_name}]: {counts[cls_name]}/{samples_per_class} saved")

    print("\n---------------- Data Fetching Summary ----------------")
    for cls_name in target_classes:
        folder = os.path.join(output_dir, cls_name)
        total_files = len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))])
        print(f"  - {cls_name:12s}: {total_files} total images in {folder}")
    print("-------------------------------------------------------\n")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download diverse hand gesture images from Hugging Face.")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory for raw images")
    parser.add_argument("--samples", type=int, default=200, help="Target samples per gesture class")
    parser.add_argument("--dataset", type=str, default="s17660101713/hagrid-subset", help="Hugging Face dataset")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download threads")
    args = parser.parse_args()

    fetch_hagrid_subset(
        output_dir=args.output_dir,
        samples_per_class=args.samples,
        dataset_name=args.dataset,
        max_workers=args.workers,
    )
