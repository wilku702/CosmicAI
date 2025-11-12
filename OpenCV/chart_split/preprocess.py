from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


def _resize_with_aspect(
    image: np.ndarray,
    max_dim: int = 768,
    pad: bool = True,
    border_color: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[np.ndarray, Dict[str, float]]:
    if max_dim <= 0:
        meta = {
            "scale": 1.0,
            "pad_top": 0,
            "pad_left": 0,
            "final_height": int(image.shape[0]),
            "final_width": int(image.shape[1]),
        }
        return image.copy(), meta

    h, w = image.shape[:2]
    max_side = float(max(h, w) or 1.0)
    scale = max_dim / max_side
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    if not pad or (new_w == max_dim and new_h == max_dim):
        meta = {
            "scale": float(scale),
            "pad_top": 0,
            "pad_left": 0,
            "final_height": int(resized.shape[0]),
            "final_width": int(resized.shape[1]),
        }
        return resized, meta

    pad_h = max_dim - new_h
    pad_w = max_dim - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    bordered = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=border_color,
    )
    meta = {
        "scale": float(scale),
        "pad_top": int(top),
        "pad_left": int(left),
        "final_height": int(bordered.shape[0]),
        "final_width": int(bordered.shape[1]),
    }
    return bordered, meta


def _apply_clahe(
    image: np.ndarray,
    clip_limit: float,
    tile_grid_size: Tuple[int, int],
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _adaptive_binary(
    gray: np.ndarray,
    block_size: int,
    c: int,
) -> np.ndarray:
    block = block_size if block_size % 2 == 1 else block_size + 1
    block = max(block, 3)
    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        c,
    )
    return cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )


def _morphological_skeleton(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(3, kernel_size | 1)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    skeleton = np.zeros_like(mask)
    working = mask.copy()

    while cv2.countNonZero(working) > 0:
        opened = cv2.morphologyEx(working, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(working, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        working = cv2.erode(working, element)

    return skeleton


def _highlight_lines(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = mask.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, int(w * 0.06)), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(9, int(h * 0.06))))
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)
    line_map = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0)
    return horizontal, vertical, line_map


def _component_stats(mask: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
    h, w = mask.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    viz = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    palette = rng.integers(0, 255, size=(max(num_labels, 2), 3), dtype=np.uint8)
    denom = float(mask.size or 1)
    details: List[dict] = []

    for label_id in range(1, num_labels):
        viz[labels == label_id] = palette[label_id]
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        left = int(stats[label_id, cv2.CC_STAT_LEFT])
        top = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_id]
        details.append(
            {
                "id": int(label_id),
                "bbox": [left, top, width, height],
                "area_ratio": float(area) / denom,
                "centroid": [float(cx) / float(w or 1), float(cy) / float(h or 1)],
            }
        )

    return viz, details


def _dominant_palette(
    image: np.ndarray,
    palette_size: int,
    max_samples: int = 120_000,
) -> List[dict]:
    if palette_size <= 0:
        return []

    pixels = image.reshape(-1, 3)
    if not len(pixels):
        return []

    if len(pixels) > max_samples:
        rng = np.random.default_rng(1)
        indices = rng.choice(len(pixels), size=max_samples, replace=False)
        samples = pixels[indices]
    else:
        samples = pixels

    k = min(palette_size, len(samples))
    samples = np.float32(samples)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
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
    return [
        {"color": [int(round(value)) for value in centers[idx]], "fraction": float(fractions[idx])}
        for idx in order
    ]


def _position_from_centroid(centroid: Tuple[float, float]) -> str:
    cx, cy = centroid
    if cy < 0.25:
        return "top-left" if cx < 0.5 else "top-right"
    if cy > 0.75:
        return "bottom-left" if cx < 0.5 else "bottom-right"
    if cx < 0.25:
        return "left"
    if cx > 0.75:
        return "right"
    return "center"


def _region_descriptors(
    component_stats: List[dict],
    image_shape: Tuple[int, int, int],
    limit: int = 24,
) -> List[dict]:
    h, w = image_shape[:2]
    descriptors: List[dict] = []
    sorted_components = sorted(
        component_stats,
        key=lambda comp: comp.get("area_ratio", 0.0),
        reverse=True,
    )
    for comp in sorted_components[:limit]:
        left, top, width, height = comp["bbox"]
        descriptors.append(
            {
                "label": "component",
                "bbox": [left, top, width, height],
                "area_ratio": comp["area_ratio"],
                "aspect_ratio": float(width) / float(height or 1),
                "center": comp["centroid"],
                "position": _position_from_centroid(tuple(comp["centroid"])),
                "height_ratio": float(height) / float(h or 1),
                "width_ratio": float(width) / float(w or 1),
            }
        )
    return descriptors


def preprocess_for_llm(
    image: np.ndarray,
    *,
    max_dim: int = 768,
    pad_to_square: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8),
    adaptive_block_size: int = 35,
    adaptive_c: int = 7,
    skeleton_kernel: int = 3,
    palette_size: int = 5,
) -> dict:
    normalized, _ = _resize_with_aspect(
        image,
        max_dim=max_dim,
        pad=pad_to_square,
        border_color=(255, 255, 255),
    )
    enhanced = _apply_clahe(
        normalized,
        clip_limit=clahe_clip_limit,
        tile_grid_size=clahe_grid,
    )
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    binary = _adaptive_binary(gray, adaptive_block_size, adaptive_c)
    edges = cv2.Canny(gray, 60, 180)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    skeleton = _morphological_skeleton(binary, skeleton_kernel)
    horizontal_lines, vertical_lines, line_map = _highlight_lines(binary)
    component_map, components = _component_stats(binary)
    palette = _dominant_palette(enhanced, palette_size=palette_size)
    regions = _region_descriptors(components, enhanced.shape)

    stats = {
        "edge_density": float(cv2.countNonZero(edges)) / float(edges.size or 1),
        "foreground_ratio": float(cv2.countNonZero(binary)) / float(binary.size or 1),
        "mean_brightness": float(gray.mean()) / 255.0,
        "contrast": float(gray.std()) / 255.0,
    }

    return {
        "normalized": normalized,
        "enhanced": enhanced,
        "binary": binary,
        "edges": edges,
        "gradient": gradient,
        "skeleton": skeleton,
        "horizontal_lines": horizontal_lines,
        "vertical_lines": vertical_lines,
        "line_map": line_map,
        "component_map": component_map,
        "stats": stats,
        "component_stats": components,
        "palette": palette,
        "region_descriptors": regions,
    }


def save_overlays(artifacts: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays = {
        "normalized.png": artifacts["normalized"],
        "enhanced.png": artifacts["enhanced"],
        "binary.png": artifacts["binary"],
        "edges.png": artifacts["edges"],
        "gradient.png": artifacts["gradient"],
        "skeleton.png": artifacts["skeleton"],
        "horizontal_lines.png": artifacts["horizontal_lines"],
        "vertical_lines.png": artifacts["vertical_lines"],
        "line_map.png": artifacts["line_map"],
        "component_map.png": artifacts["component_map"],
    }
    for name, data in overlays.items():
        cv2.imwrite(str(output_dir / name), data)


__all__ = [
    "preprocess_for_llm",
    "save_overlays",
]
