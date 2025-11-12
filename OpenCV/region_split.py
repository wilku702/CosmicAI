"""
Utilities for splitting chart/plot images into semantically meaningful regions.

The module provides two complementary segmentation strategies and helper
functions to merge and visualize the resulting regions so they can be consumed
by downstream visual-language models.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Region:
    """Container for rectangular regions."""

    bbox: Tuple[int, int, int, int]
    label: str = "region"
    score: float = 1.0

    def area(self) -> int:
        """Return the area of the region."""
        x, y, w, h = self.bbox
        return w * h

    def iou(self, other: "Region") -> float:
        """Compute intersection over union with another region."""
        ax, ay, aw, ah = self.bbox
        bx, by, bw, bh = other.bbox

        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax + aw, bx + bw)
        iy2 = min(ay + ah, by + bh)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih

        if inter == 0:
            return 0.0

        union = self.area() + other.area() - inter
        return inter / float(union)

    def merge(self, other: "Region", label: str | None = None) -> "Region":
        """Return a new region covering both regions."""
        ax, ay, aw, ah = self.bbox
        bx, by, bw, bh = other.bbox

        x1 = min(ax, bx)
        y1 = min(ay, by)
        x2 = max(ax + aw, bx + bw)
        y2 = max(ay + ah, by + bh)

        merged_label = label if label else f"{self.label}+{other.label}"
        merged_score = (self.score + other.score) / 2.0
        return Region((x1, y1, x2 - x1, y2 - y1), merged_label, merged_score)


def _append_region(
    regions: List[Region],
    candidate: Region,
    iou_threshold: float = 0.85,
) -> None:
    """
    Append a region unless a similar one with the same label already exists.
    """
    for idx, existing in enumerate(regions):
        if existing.label != candidate.label:
            continue
        if existing.iou(candidate) >= iou_threshold:
            if candidate.score > existing.score:
                regions[idx] = candidate
            return
    regions.append(candidate)


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _ensure_color(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    gray = _ensure_gray(image)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _odd_kernel(size: int) -> int:
    size = int(size)
    return size if size % 2 == 1 else size + 1


def structured_split(
    image: np.ndarray,
    blur_kernel: int = 5,
    canny_threshold1: int = 50,
    canny_threshold2: int = 180,
    min_bar_area_ratio: float = 0.01,
    rectangularity_threshold: float = 0.75,
    legend_area_ratio: float = 0.005,
    axis_min_length_ratio: float = 0.45,
    hough_vote_threshold: int = 120,
    line_kernel_ratio: float = 0.025,
    min_line_area_ratio: float = 0.0005,
) -> List[Region]:
    """
    Detect structured chart elements such as bars, lines, axes, and legends.

    Parameters
    ----------
    image:
        Input RGB or grayscale image.
    blur_kernel:
        Gaussian blur kernel size for noise suppression.
    canny_threshold1, canny_threshold2:
        Lower/upper thresholds for the Canny edge detector.
    min_bar_area_ratio:
        Minimum contour area relative to the whole image to consider a bar.
    rectangularity_threshold:
        Minimum contour area divided by its bounding rectangle area to keep.
    legend_area_ratio:
        Maximum area ratio for candidate legend boxes.
    axis_min_length_ratio:
        Minimum normalized length (relative to image diagonal) for axes/lines.
    hough_vote_threshold:
        Votes required by the probabilistic Hough transform to accept a line.
    line_kernel_ratio:
        Relative size of the morphology kernel used to emphasize thin lines.
    min_line_area_ratio:
        Minimum normalized area for morphology-derived line candidates.
    """
    color_image = _ensure_color(image)
    gray = _ensure_gray(image)
    h, w = gray.shape
    image_area = float(h * w)
    diag = float(np.hypot(w, h))

    blur_k = _odd_kernel(blur_kernel)
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    _, binary_inv = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary = cv2.bitwise_not(binary_inv)

    regions: List[Region] = []
    approx_scale = 0.02
    for mask in (binary_inv, binary):
        contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_bar_area_ratio * image_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            epsilon = approx_scale * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            x, y, width, height = cv2.boundingRect(approx)
            if width == 0 or height == 0:
                continue

            rectangularity = area / float(width * height)
            aspect_ratio = width / float(height)

            if rectangularity >= rectangularity_threshold and len(approx) >= 4:
                label = "bar"
                score = min(rectangularity, 1.0)

                if area <= legend_area_ratio * image_area:
                    label = "legend"
                    score *= 0.9

                _append_region(
                    regions, Region((x, y, width, height), label, score), iou_threshold=0.8
                )
            elif aspect_ratio >= 4.0 or aspect_ratio <= 0.25:
                label = "axis"
                _append_region(
                    regions, Region((x, y, width, height), label, 0.6), iou_threshold=0.7
                )

    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

    line_regions: List[Region] = []
    lines = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=hough_vote_threshold,
        minLineLength=int(axis_min_length_ratio * diag),
        maxLineGap=12,
    )

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < axis_min_length_ratio * diag:
                continue

            pad = 6
            x_min = max(0, min(x1, x2) - pad)
            y_min = max(0, min(y1, y2) - pad)
            x_max = min(w, max(x1, x2) + pad)
            y_max = min(h, max(y1, y2) + pad)

            is_axis = (
                min(y1, y2) < int(0.05 * h)
                or max(y1, y2) > int(0.95 * h)
                or min(x1, x2) < int(0.05 * w)
                or max(x1, x2) > int(0.95 * w)
            )
            label = "axis" if is_axis else "line"
            score = 0.9 if label == "line" else 0.7
            _append_region(
                line_regions,
                Region((x_min, y_min, x_max - x_min, y_max - y_min), label, score),
                iou_threshold=0.65,
            )

    kernel_len = max(3, int(min(h, w) * line_kernel_ratio))
    if kernel_len >= 3:
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))

        def _extract_morphological_lines(
            mask: np.ndarray, kernel: np.ndarray, orientation: str
        ) -> None:
            morphed = cv2.erode(mask, kernel, iterations=1)
            morphed = cv2.dilate(morphed, kernel, iterations=1)
            contours, _ = cv2.findContours(
                morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                if width == 0 or height == 0:
                    continue
                length = width if orientation == "horizontal" else height
                supporting_span = w if orientation == "horizontal" else h
                if length < axis_min_length_ratio * supporting_span:
                    continue
                if width * height < min_line_area_ratio * image_area:
                    continue

                is_axis = (
                    min(y, y + height) < int(0.05 * h)
                    or max(y, y + height) > int(0.95 * h)
                    or min(x, x + width) < int(0.05 * w)
                    or max(x, x + width) > int(0.95 * w)
                )
                label = "axis" if is_axis else "line"
                normalized_length = min(1.0, length / (diag or 1.0))
                score = 0.75 + 0.2 * normalized_length
                _append_region(
                    line_regions,
                    Region((x, y, width, height), label, score),
                    iou_threshold=0.65,
                )

        for mask in (binary_inv, binary):
            _extract_morphological_lines(mask, horizontal_kernel, "horizontal")
            _extract_morphological_lines(mask, vertical_kernel, "vertical")

    regions.extend(line_regions)
    return regions


def unstructured_split(
    image: np.ndarray,
    morph_kernel: int = 3,
    distance_kernel: int = 5,
    distance_ratio: float = 0.35,
    min_area_ratio: float = 0.002,
) -> List[Region]:
    """
    Split touching or overlapping regions via marker-based watershed.

    The implementation follows the classical pipeline of noise removal,
    foreground extraction, marker generation, and watershed expansion
    [oai_citation:0‡GeeksforGeeks](https://www.geeksforgeeks.org/computer-vision/image-segmentation-with-watershed-algorithm-opencv-python/?utm_source=chatgpt.com).

    Parameters
    ----------
    image:
        Input RGB or grayscale image.
    morph_kernel:
        Size of the morphological kernel used for opening/dilation.
    distance_kernel:
        Kernel size for the distance transform.
    distance_ratio:
        Threshold ratio applied on the distance map to pick sure foreground.
    min_area_ratio:
        Minimum normalized area to keep a final region.
    """
    color_image = _ensure_color(image)
    gray = _ensure_gray(color_image)
    h, w = gray.shape
    image_area = float(h * w)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (int(morph_kernel), int(morph_kernel))
    )
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)

    _, thresh = cv2.threshold(
        opening, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    sure_bg = cv2.dilate(thresh, kernel, iterations=2)
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, int(distance_kernel))

    _, sure_fg = cv2.threshold(
        dist,
        distance_ratio * dist.max(),
        255,
        0,
    )

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    cv2.watershed(color_image, markers)

    regions: List[Region] = []
    for marker_id in range(2, num_markers + 2):
        mask = np.uint8(markers == marker_id) * 255
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area_ratio * image_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            regions.append(Region((x, y, width, height), "segment", 0.8))

    return regions


def _normalized_center(region: Region) -> Tuple[float, float]:
    x, y, w, h = region.bbox
    return (x + w / 2.0, y + h / 2.0)


def merge_regions(
    regions: Sequence[Region],
    max_regions: int,
    frame_shape: Tuple[int, int] | None = None,
    iou_threshold: float = 0.4,
    center_distance_threshold: float = 0.12,
    alignment_tolerance: float = 0.18,
) -> List[Region]:
    """
    Merge regions based on IoU, center distance, and axis alignment.

    Parameters
    ----------
    regions:
        Regions to merge.
    max_regions:
        Desired budget (stop once the number of regions is under this limit).
    frame_shape:
        Image (height, width). Used to normalize distances; inferred when absent.
    iou_threshold:
        Minimum IoU to merge on overlap.
    center_distance_threshold:
        Maximum normalized center distance to merge non-overlapping boxes.
    alignment_tolerance:
        Maximum normalized offset for horizontal/vertical alignment merging.
    """
    merged = list(regions)
    if len(merged) <= max_regions:
        return merged

    if frame_shape is None:
        frame_shape = (0, 0)

    height, width = frame_shape
    diag = float(np.hypot(width, height)) if width and height else 1.0

    def normalized_distance(a: Region, b: Region) -> float:
        ax, ay = _normalized_center(a)
        bx, by = _normalized_center(b)
        return float(np.hypot(ax - bx, ay - by)) / diag

    while len(merged) > max_regions:
        best_pair = None
        best_score = -np.inf

        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                region_a = merged[i]
                region_b = merged[j]

                iou = region_a.iou(region_b)
                dist = normalized_distance(region_a, region_b)

                ax, ay, aw, ah = region_a.bbox
                bx, by, bw, bh = region_b.bbox

                vertical_alignment = abs((ay + ah / 2.0) - (by + bh / 2.0)) / (
                    diag or 1.0
                )
                horizontal_alignment = abs((ax + aw / 2.0) - (bx + bw / 2.0)) / (
                    diag or 1.0
                )

                aligned = (
                    vertical_alignment < alignment_tolerance
                    or horizontal_alignment < alignment_tolerance
                )

                local_score = (
                    2.0 * iou
                    + (1.0 - dist if dist < center_distance_threshold else 0.0)
                    + (0.5 if aligned else 0.0)
                )

                if iou < iou_threshold and not aligned and dist >= center_distance_threshold:
                    continue

                if local_score > best_score:
                    best_score = local_score
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair
        merged_region = merged[i].merge(merged[j])
        merged.pop(j)
        merged.pop(i)
        merged.append(merged_region)

    return merged


def draw_regions(
    image: np.ndarray,
    regions: Sequence[Region],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    copy: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes with labels on top of the input image.

    Parameters
    ----------
    image:
        Input RGB/BGR image.
    regions:
        Regions to visualize.
    color:
        BGR color for the bounding boxes.
    thickness:
        Line thickness for the rectangles.
    copy:
        When True, operate on a copy and keep the original untouched.
    """
    output = _ensure_color(image)
    if copy:
        output = output.copy()

    for region in regions:
        x, y, w, h = region.bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), color, int(thickness))
        cv2.putText(
            output,
            f"{region.label}:{region.score:.2f}",
            (x, max(y - 4, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    return output


def overlay_masks(
    image: np.ndarray,
    masks: Iterable[np.ndarray],
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay binary masks in random colors to inspect segmentation quality.

    Parameters
    ----------
    image:
        Input RGB/BGR image.
    masks:
        Iterable of binary masks aligned with the image.
    alpha:
        Opacity of the colored overlays.
    """
    output = _ensure_color(image).astype(np.float32)

    for mask in masks:
        if mask.shape[:2] != output.shape[:2]:
            raise ValueError("Mask shape must match the image resolution.")
        color = np.random.default_rng().integers(0, 255, size=3, dtype=np.uint8)
        colored_mask = cv2.merge(
            [
                np.where(mask > 0, int(color[0]), 0).astype(np.uint8),
                np.where(mask > 0, int(color[1]), 0).astype(np.uint8),
                np.where(mask > 0, int(color[2]), 0).astype(np.uint8),
            ]
        )
        output = cv2.addWeighted(output, 1.0, colored_mask.astype(np.float32), alpha, 0)

    return np.clip(output, 0, 255).astype(np.uint8)


SUPPORTED_IMAGE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _resize_with_aspect(
    image: np.ndarray,
    max_dim: int = 768,
    pad: bool = True,
    border_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Resize image so its longest side equals ``max_dim`` while keeping aspect ratio.
    Optionally pad to obtain a square canvas which is easier for downstream models.
    """
    if max_dim <= 0:
        return image.copy()

    h, w = image.shape[:2]
    max_side = float(max(h, w))
    if max_side == 0:
        return image.copy()

    scale = max_dim / max_side
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    if not pad:
        return resized

    pad_h = max(0, max_dim - new_h)
    pad_w = max(0, max_dim - new_w)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if pad_h == 0 and pad_w == 0:
        return resized

    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=border_color,
    )


def _apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _extract_color_palette(
    image: np.ndarray,
    palette_size: int = 5,
    max_samples: int = 120_000,
) -> List[dict]:
    """
    Return the dominant colors with their relative coverage ratios.
    """
    if palette_size <= 0:
        return []

    pixels = image.reshape(-1, 3)
    total_pixels = pixels.shape[0]
    if total_pixels == 0:
        return []

    if total_pixels > max_samples:
        rng = np.random.default_rng(0)
        indices = rng.choice(total_pixels, size=max_samples, replace=False)
        samples = pixels[indices]
    else:
        samples = pixels

    k = min(palette_size, samples.shape[0])
    samples = np.float32(samples)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    compactness, labels, centers = cv2.kmeans(
        samples,
        k,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    counts = np.bincount(labels.flatten(), minlength=k).astype(np.float32)
    fractions = counts / float(counts.sum() or 1.0)

    order = np.argsort(fractions)[::-1]
    palette = []
    for idx in order:
        color = [int(round(value)) for value in centers[idx]]
        palette.append({"color": color, "fraction": float(fractions[idx])})

    return palette


def _morphological_skeleton(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel_size = max(3, kernel_size | 1)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    skeleton = np.zeros_like(binary)
    working = binary.copy()

    while True:
        opened = cv2.morphologyEx(working, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(working, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        working = cv2.erode(working, element)
        if cv2.countNonZero(working) == 0:
            break

    return skeleton


def preprocess_for_llm(
    image: np.ndarray,
    *,
    regions: Sequence[Region] | None = None,
    max_dim: int = 768,
    pad_to_square: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8),
    palette_size: int = 5,
    adaptive_block_size: int = 35,
    adaptive_c: int = 7,
    skeleton_kernel: int = 3,
) -> dict:
    """
    Produce normalized projections that make it easier for an LLM to critique charts.
    """
    normalized = _resize_with_aspect(
        image, max_dim=max_dim, pad=pad_to_square, border_color=(255, 255, 255)
    )
    enhanced = _apply_clahe(
        normalized, clip_limit=clahe_clip_limit, tile_grid_size=clahe_grid
    )
    gray = _ensure_gray(enhanced)

    block = adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        max(block, 3),
        adaptive_c,
    )
    cleaned = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    edges = cv2.Canny(gray, 60, 180)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    skeleton = _morphological_skeleton(cleaned, kernel_size=skeleton_kernel)
    palette = _extract_color_palette(enhanced, palette_size=palette_size)

    stats = {
        "edge_density": float(cv2.countNonZero(edges)) / float(edges.size or 1),
        "foreground_ratio": float(cv2.countNonZero(cleaned)) / float(cleaned.size or 1),
        "mean_brightness": float(gray.mean()) / 255.0,
        "contrast": float(gray.std()) / 255.0,
    }

    if regions:
        label_counts = {}
        for region in regions:
            label_counts[region.label] = label_counts.get(region.label, 0) + 1
        stats["region_counts"] = label_counts
        stats["region_budget"] = len(regions)

    return {
        "normalized": normalized,
        "enhanced": enhanced,
        "binary": cleaned,
        "edges": edges,
        "gradient": gradient,
        "skeleton": skeleton,
        "palette": palette,
        "stats": stats,
    }


def _save_region_crops(
    image: np.ndarray,
    regions: Sequence[Region],
    target_dir: Path,
    max_crops: int = 12,
    padding: int = 4,
) -> List[dict]:
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: List[dict] = []
    sorted_regions = sorted(regions, key=lambda reg: reg.area(), reverse=True)

    for idx, region in enumerate(sorted_regions[:max(0, max_crops)]):
        x, y, w, h = region.bbox
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(image.shape[1], x + w + padding)
        y1 = min(image.shape[0], y + h + padding)
        crop = image[y0:y1, x0:x1]
        filename = f"{idx:02d}_{region.label}.png"
        cv2.imwrite(str(target_dir / filename), crop)
        saved.append({"file": filename, "label": region.label, "score": float(region.score)})

    return saved


def _write_llm_preprocess_assets(
    image: np.ndarray,
    structured: Sequence[Region],
    unstructured: Sequence[Region],
    merged: Sequence[Region],
    output_root: Path,
    image_id: str,
    *,
    max_dim: int = 768,
    palette_size: int = 5,
    adaptive_block_size: int = 35,
    adaptive_c: int = 7,
    skeleton_kernel: int = 3,
    max_region_crops: int = 12,
) -> None:
    artifacts = preprocess_for_llm(
        image,
        regions=merged,
        max_dim=max_dim,
        palette_size=palette_size,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
        skeleton_kernel=skeleton_kernel,
    )

    target = output_root / image_id
    target.mkdir(parents=True, exist_ok=True)

    projections = {
        "normalized.png": artifacts["normalized"],
        "enhanced.png": artifacts["enhanced"],
        "binary.png": artifacts["binary"],
        "edges.png": artifacts["edges"],
        "gradient.png": artifacts["gradient"],
        "skeleton.png": artifacts["skeleton"],
    }
    for name, data in projections.items():
        cv2.imwrite(str(target / name), data)

    crop_info = []
    if merged:
        crop_info = _save_region_crops(
            image,
            merged,
            target / "crops",
            max_crops=max_region_crops,
            padding=6,
        )

    metadata = {
        "palette": artifacts["palette"],
        "stats": artifacts["stats"],
        "structured_regions": len(structured),
        "unstructured_regions": len(unstructured),
        "merged_regions": len(merged),
        "region_crops": crop_info,
    }

    with (target / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


def _collect_image_files(image_dir: str, suffixes: Sequence[str] = SUPPORTED_IMAGE_SUFFIXES) -> List[Path]:
    """
    Return a sorted list of image file paths that match the supported suffixes.
    """
    base_dir = Path(image_dir).expanduser()
    if not base_dir.exists():
        raise FileNotFoundError(f"Image directory '{image_dir}' does not exist.")
    if base_dir.is_file() and base_dir.suffix.lower() in suffixes:
        return [base_dir]
    files = sorted(
        path
        for path in base_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    if not files:
        raise FileNotFoundError(
            f"Found no files with suffixes {', '.join(suffixes)} under '{image_dir}'."
        )
    return files


def _regions_to_dict(regions: Sequence[Region]) -> List[dict]:
    return [
        {
            "bbox": [int(coord) for coord in region.bbox],
            "label": str(region.label),
            "score": float(region.score),
        }
        for region in regions
    ]


def _write_debug_outputs(
    image: np.ndarray,
    structured: Sequence[Region],
    unstructured: Sequence[Region],
    merged: Sequence[Region],
    output_root: Path,
    image_id: str,
    save_visuals: bool,
) -> None:
    """
    Persist region metadata and optional visualizations for a single image.
    """
    target = output_root / image_id
    target.mkdir(parents=True, exist_ok=True)

    payload = {
        "structured": _regions_to_dict(structured),
        "unstructured": _regions_to_dict(unstructured),
        "merged": _regions_to_dict(merged),
    }
    with (target / "regions.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    if not save_visuals:
        return

    variants = {
        "structured": (structured, (0, 255, 0)),
        "unstructured": (unstructured, (255, 165, 0)),
        "merged": (merged, (0, 128, 255)),
    }
    for name, (regions, color) in variants.items():
        debug = draw_regions(image, regions, color=color, copy=True)
        cv2.imwrite(str(target / f"{name}.png"), debug)


def run_graph_image_tests(
    image_dir: str,
    output_dir: str | None = None,
    max_regions: int = 12,
    limit: int | None = None,
    save_visuals: bool = True,
    verbose: bool = True,
    preprocess_dir: str | None = None,
    preprocess_max_dim: int = 768,
    palette_size: int = 5,
    adaptive_block_size: int = 35,
    adaptive_c: int = 7,
    skeleton_kernel: int = 3,
    max_region_crops: int = 12,
) -> List[dict]:
    """
    Execute both segmentation strategies on all images inside ``image_dir``.

    Parameters
    ----------
    image_dir:
        Directory that contains graph/chart images.
    output_dir:
        Optional directory where debug visualizations and JSON metadata are saved.
    max_regions:
        Budget passed to ``merge_regions`` when combining the two splitters.
    limit:
        When provided, restrict the number of processed images.
    save_visuals:
        Toggle saving rendered overlays when ``output_dir`` is provided.
    verbose:
        Print per-image statistics when True.
    preprocess_dir:
        Optional directory in which richer preprocessing artifacts (binary maps,
        edges, skeletons, palettes) are written for LLM consumption.
    preprocess_max_dim:
        Maximum canvas size when normalizing charts for preprocessing.
    palette_size:
        Number of dominant colors to summarize per chart.
    adaptive_block_size, adaptive_c:
        Parameters controlling the adaptive threshold inside preprocessing.
    skeleton_kernel:
        Kernel size used when extracting chart skeletons.
    max_region_crops:
        Number of top regions to export as cropped patches during preprocessing.
    """
    image_paths = _collect_image_files(image_dir)
    if limit is not None and limit > 0:
        image_paths = image_paths[:limit]
    total = len(image_paths)

    output_root = Path(output_dir).expanduser() if output_dir else None
    if output_root:
        output_root.mkdir(parents=True, exist_ok=True)

    preprocess_root = Path(preprocess_dir).expanduser() if preprocess_dir else None
    if preprocess_root:
        preprocess_root.mkdir(parents=True, exist_ok=True)

    results: List[dict] = []
    for idx, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            if verbose:
                print(f"[{idx}/{total}] Skipping '{image_path}' (failed to load).")
            continue

        structured = structured_split(image)
        unstructured = unstructured_split(image)
        merged = merge_regions(
            structured + unstructured,
            max_regions=max_regions,
            frame_shape=image.shape[:2],
        )

        h, w = image.shape[:2]
        area = float(max(h * w, 1))
        stats = {
            "image": str(image_path),
            "structured_regions": len(structured),
            "unstructured_regions": len(unstructured),
            "merged_regions": len(merged),
            "structured_coverage": sum(region.area() for region in structured) / area,
            "unstructured_coverage": sum(region.area() for region in unstructured) / area,
            "merged_coverage": sum(region.area() for region in merged) / area,
        }
        results.append(stats)

        if verbose:
            print(
                f"[{idx}/{total}] {image_path.name}: "
                f"{stats['structured_regions']} structured, "
                f"{stats['unstructured_regions']} unstructured, "
                f"{stats['merged_regions']} merged"
            )

        if output_root:
            _write_debug_outputs(
                image,
                structured,
                unstructured,
                merged,
                output_root,
                image_path.stem,
                save_visuals=save_visuals,
            )

        if preprocess_root:
            _write_llm_preprocess_assets(
                image,
                structured,
                unstructured,
                merged,
                preprocess_root,
                image_path.stem,
                max_dim=preprocess_max_dim,
                palette_size=palette_size,
                adaptive_block_size=adaptive_block_size,
                adaptive_c=adaptive_c,
                skeleton_kernel=skeleton_kernel,
                max_region_crops=max_region_crops,
            )

    if output_root:
        with (output_root / "summary.json").open("w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2)

    return results


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run the chart segmentation pipeline on a folder of graph images."
    )
    parser.add_argument(
        "image_dir",
        help="Path to the directory that stores the graph/chart images.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_dir",
        default=None,
        help="Directory to store overlays and JSON summaries.",
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=30,
        help="Maximum number of regions when merging both strategies.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N images.",
    )
    parser.add_argument(
        "--no-visuals",
        action="store_false",
        dest="save_visuals",
        help="Skip writing overlay PNGs (still writes JSON summaries).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-image logging.",
    )
    parser.add_argument(
        "--preprocess-dir",
        default=None,
        help="Directory to store LLM-oriented preprocessing artifacts.",
    )
    parser.add_argument(
        "--preprocess-max-dim",
        type=int,
        default=768,
        help="Canvas size for normalization before feeding charts to the LLM.",
    )
    parser.add_argument(
        "--palette-size",
        type=int,
        default=5,
        help="Number of dominant colors to extract per chart.",
    )
    parser.add_argument(
        "--adaptive-block-size",
        type=int,
        default=35,
        help="Adaptive threshold block size used during preprocessing.",
    )
    parser.add_argument(
        "--adaptive-c",
        type=int,
        default=7,
        help="Adaptive threshold constant used during preprocessing.",
    )
    parser.add_argument(
        "--skeleton-kernel",
        type=int,
        default=3,
        help="Kernel size for morphological skeleton extraction.",
    )
    parser.add_argument(
        "--region-crops",
        type=int,
        default=12,
        help="Number of top merged regions to export as crops.",
    )
    args = parser.parse_args()

    run_graph_image_tests(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        max_regions=args.max_regions,
        limit=args.limit,
        save_visuals=args.save_visuals,
        verbose=not args.quiet,
        preprocess_dir=args.preprocess_dir,
        preprocess_max_dim=args.preprocess_max_dim,
        palette_size=args.palette_size,
        adaptive_block_size=args.adaptive_block_size,
        adaptive_c=args.adaptive_c,
        skeleton_kernel=args.skeleton_kernel,
        max_region_crops=args.region_crops,
    )


if __name__ == "__main__":
    _cli()
