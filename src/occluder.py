"""Synthetic occlusion generator using OpenCV cutout/block simulation."""

from typing import Optional, Tuple, Literal
import os
import random
import numpy as np
import cv2


def apply_block_occlusion(
    image: np.ndarray,
    occlusion_pct: float,
    occlude_roi: str = "random_hand_patch",
    hand_bbox: Optional[Tuple[int, int, int, int]] = None,
    occlusion_type: Literal["black", "noise", "gray", "blur"] = "black",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate real-world contiguous physical blockage on an image.

    Args:
        image: Original BGR image (H, W, 3).
        occlusion_pct: Fraction of hand bounding box area to obscure [0.0, 1.0].
                       E.g., 0.0, 0.10, 0.20, 0.30, 0.45, 0.60.
        occlude_roi: Strategy for placement ('random_hand_patch', 'center_hand', 'frame').
        hand_bbox: Optional (xmin, ymin, xmax, ymax) pixel bounding box of hand.
        occlusion_type: Type of occlusion block ('black', 'noise', 'gray', 'blur').
        seed: Random seed for reproducibility.

    Returns:
        Occluded BGR image.
    """
    if occlusion_pct <= 0.0 or image is None:
        return image.copy()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    h_img, w_img = image.shape[:2]
    occluded_img = image.copy()

    # Determine reference bounding box
    if hand_bbox is not None:
        xmin, ymin, xmax, ymax = hand_bbox
        # Ensure within image boundaries
        xmin, xmax = max(0, xmin), min(w_img, xmax)
        ymin, ymax = max(0, ymin), min(h_img, ymax)
    else:
        # Default fallback to center 60% of frame as approximate hand region
        box_w = int(w_img * 0.6)
        box_h = int(h_img * 0.6)
        xmin = (w_img - box_w) // 2
        ymin = (h_img - box_h) // 2
        xmax = xmin + box_w
        ymax = ymin + box_h

    roi_w = max(10, xmax - xmin)
    roi_h = max(10, ymax - ymin)
    roi_area = roi_w * roi_h

    # Target area of occlusion
    target_block_area = max(1.0, occlusion_pct * roi_area)

    # Pick an aspect ratio between 0.75 and 1.33
    aspect_ratio = random.uniform(0.75, 1.33)
    block_w = int(np.clip(np.sqrt(target_block_area * aspect_ratio), 1, w_img))
    block_h = int(np.clip(target_block_area / max(1, block_w), 1, h_img))

    # Determine top-left corner (x1, y1)
    if occlude_roi == "center_hand":
        cx = xmin + roi_w // 2
        cy = ymin + roi_h // 2
        x1 = cx - block_w // 2
        y1 = cy - block_h // 2
    elif occlude_roi == "random_hand_patch":
        # Anchor the block to overlap the hand region
        min_x = max(0, xmin - block_w // 3)
        max_x = min(w_img - block_w, xmax - (2 * block_w) // 3)
        min_y = max(0, ymin - block_h // 3)
        max_y = min(h_img - block_h, ymax - (2 * block_h) // 3)

        x1 = random.randint(min_x, max(min_x, max_x)) if max_x >= min_x else xmin
        y1 = random.randint(min_y, max(min_y, max_y)) if max_y >= min_y else ymin
    else:
        # Full frame placement
        x1 = random.randint(0, max(0, w_img - block_w))
        y1 = random.randint(0, max(0, h_img - block_h))

    # Keep coordinates strictly within image boundaries
    x1 = int(np.clip(x1, 0, w_img - 1))
    y1 = int(np.clip(y1, 0, h_img - 1))
    x2 = int(np.clip(x1 + block_w, x1 + 1, w_img))
    y2 = int(np.clip(y1 + block_h, y1 + 1, h_img))

    # Apply the occlusion block
    if occlusion_type == "black":
        occluded_img[y1:y2, x1:x2] = 0
    elif occlusion_type == "gray":
        occluded_img[y1:y2, x1:x2] = 128
    elif occlusion_type == "noise":
        noise = np.random.randint(0, 256, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
        occluded_img[y1:y2, x1:x2] = noise
    elif occlusion_type == "blur":
        sub = occluded_img[y1:y2, x1:x2]
        ksize = max(15, (min(y2 - y1, x2 - x1) // 2) * 2 + 1)
        occluded_img[y1:y2, x1:x2] = cv2.GaussianBlur(sub, (ksize, ksize), 30)

    return occluded_img


def save_occlusion_preview(
    clean_image: np.ndarray,
    output_dir: str = "data/occluded",
    filename_prefix: str = "sample",
    hand_bbox: Optional[Tuple[int, int, int, int]] = None,
    percentages=(0.0, 0.10, 0.20, 0.30, 0.45, 0.60),
) -> None:
    """Generate and save inspection images across all standard occlusion thresholds."""
    os.makedirs(output_dir, exist_ok=True)
    for pct in percentages:
        occ_img = apply_block_occlusion(
            image=clean_image,
            occlusion_pct=pct,
            hand_bbox=hand_bbox,
            occlusion_type="black",
        )
        # Put text overlay indicating occlusion level
        label = f"Occlusion: {int(pct * 100)}%"
        cv2.putText(
            occ_img,
            label,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        out_path = os.path.join(output_dir, f"{filename_prefix}_occ_{int(pct * 100)}pct.jpg")
        cv2.imwrite(out_path, occ_img)
