from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .preprocess import preprocess_for_llm
from .spec_schema import ExpectedSpec, from_json_file


def _infer_chart_type(component_stats: List[dict], line_map: np.ndarray) -> str:
    tall_rects = 0
    for comp in component_stats:
        left, top, width, height = comp["bbox"]
        if width <= 0 or height <= 0:
            continue
        if height > width * 1.6 and width > 4:
            if top + height > int(0.55 * line_map.shape[0]):
                tall_rects += 1
    if tall_rects >= 3:
        return "bar"

    line_pixels = int(cv2.countNonZero(line_map))
    if line_pixels / float(line_map.size or 1) > 0.015:
        return "line"

    return "other"


def _legend_position_from_centroid(cx: float, cy: float) -> str:
    if cy < 0.25:
        return "top-left" if cx < 0.5 else "top-right"
    if cy > 0.75:
        return "bottom-left" if cx < 0.5 else "bottom-right"
    if cx < 0.25:
        return "left"
    if cx > 0.75:
        return "right"
    return "inside"


def _infer_legend(component_stats: List[dict]) -> Tuple[bool, str | None]:
    for comp in component_stats:
        area_ratio = comp.get("area_ratio", 0.0)
        if not 0.0005 <= area_ratio <= 0.08:
            continue
        cx, cy = comp["centroid"]
        if cy < 0.2 or cy > 0.8 or cx < 0.2 or cx > 0.8:
            return True, _legend_position_from_centroid(cx, cy)
    return False, None


def _infer_series_count(
    chart_type: str,
    palette: List[dict],
    line_map: np.ndarray,
    component_stats: List[dict],
) -> int:
    if chart_type == "line":
        binary = np.where(line_map > 0, 255, 0).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        series = sum(1 for idx in range(1, num_labels) if stats[idx, cv2.CC_STAT_AREA] > 40)
        return max(series, 1)
    if chart_type == "bar":
        colors = {
            tuple(int(channel) for channel in entry["color"])
            for entry in palette
            if "color" in entry
        }
        if colors:
            return max(1, min(len(colors), 6))
        columns = sum(
            1
            for comp in component_stats
            if comp["bbox"][3] > comp["bbox"][2] * 1.2 and comp["bbox"][2] > 6
        )
        return max(columns // 2, 1)
    return max(len(palette), 1) if palette else 1


def _count_ticks(mask: np.ndarray, axis: str) -> int:
    h, w = mask.shape
    if axis == "x":
        strip = mask[int(0.7 * h) : int(0.95 * h), :]
        working = strip.copy()
        cut = max(1, int(0.12 * working.shape[0]))
        working[-cut:, :] = 0
        counts = (working > 0).sum(axis=0)
    else:
        strip = mask[:, : int(0.2 * w)]
        working = strip.copy()
        cut = max(1, int(0.12 * working.shape[1]))
        working[:, :cut] = 0
        counts = (working > 0).sum(axis=1)

    threshold = 8 if axis == "x" else 8
    mask_hits = counts >= threshold
    count = 0
    in_run = False
    for active in mask_hits:
        if active and not in_run:
            count += 1
            in_run = True
        elif not active:
            in_run = False
    return count


def _infer_axes(component_stats: List[dict], binary_mask: np.ndarray) -> dict:
    h, w = binary_mask.shape
    x_ticks = _count_ticks(binary_mask, "x")
    y_ticks = _count_ticks(binary_mask, "y")
    x_label_presence = False
    y_label_presence = False

    if component_stats:
        x_region = binary_mask[int(0.85 * h) : h, :]
        y_region = binary_mask[int(0.3 * h) : int(0.8 * h), int(0.06 * w) : int(0.18 * w)]
        x_label_presence = float(cv2.countNonZero(x_region)) / float(x_region.size or 1) > 0.02
        y_label_presence = float(cv2.countNonZero(y_region)) / float(y_region.size or 1) > 0.08

    return {
        "x_label_present": bool(x_label_presence),
        "y_label_present": bool(y_label_presence),
        "x_tick_count_est": x_ticks,
        "y_tick_count_est": y_ticks,
    }


def observe_generated_chart(img_path: str, artifacts: dict | None = None) -> dict:
    if artifacts is None:
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Unable to load generated chart: {img_path}")
        artifacts = preprocess_for_llm(image)
    component_stats = artifacts["component_stats"]
    palette = artifacts["palette"]
    line_map = artifacts["line_map"]
    chart_type = _infer_chart_type(component_stats, line_map)
    legend_present, legend_position = _infer_legend(component_stats)
    series_count = _infer_series_count(chart_type, palette, line_map, component_stats)
    axes = _infer_axes(component_stats, artifacts["binary"])

    return {
        "palette": palette,
        "stats": artifacts["stats"],
        "component_stats": component_stats,
        "region_descriptors": artifacts["region_descriptors"],
        "inferred": {
            "chart_type": chart_type,
            "legend_present": legend_present,
            "legend_position": legend_position,
            "series_count": series_count,
            "axes": axes,
        },
    }


def _normalize_palette_entries(palette: List[str] | None) -> List[str]:
    if not palette:
        return []
    normalized = []
    for entry in palette:
        token = entry.strip().lower()
        if token.startswith("#") and len(token) in {4, 7}:
            normalized.append(token)
        else:
            normalized.append(token)
    return normalized


def _palette_from_observation(palette: List[dict]) -> List[str]:
    normalized = []
    for entry in palette:
        color = entry.get("color")
        if not color:
            continue
        b, g, r = color
        normalized.append(f"#{int(r):02x}{int(g):02x}{int(b):02x}")
    return normalized


def compare_spec_to_observation(spec: ExpectedSpec, obs: dict) -> dict:
    issues: List[str] = []
    deltas: Dict[str, object] = {}
    inferred = obs["inferred"]

    if spec.chart_type != "other" and inferred["chart_type"] != spec.chart_type:
        issues.append("wrong_chart_type")

    if spec.legend.present is True and not inferred["legend_present"]:
        issues.append("missing_legend")
    elif spec.legend.present is False and inferred["legend_present"]:
        issues.append("legend_should_be_absent")

    if spec.legend.position and inferred["legend_position"]:
        if spec.legend.position != inferred["legend_position"]:
            issues.append("wrong_legend_position")
            deltas["legend_position"] = {
                "expected": spec.legend.position,
                "observed": inferred["legend_position"],
            }

    if spec.series.count is not None:
        diff = abs(spec.series.count - inferred["series_count"])
        if diff > 0:
            issues.append("wrong_series_count")
            deltas["series_count"] = {"expected": spec.series.count, "observed": inferred["series_count"]}

    spec_palette = _normalize_palette_entries(spec.palette)
    obs_palette = _palette_from_observation(obs["palette"])
    if spec_palette and obs_palette:
        set_spec = set(spec_palette)
        set_obs = set(obs_palette)
        jaccard = len(set_spec & set_obs) / float(len(set_spec | set_obs) or 1)
        deltas["palette_jaccard"] = jaccard
        if jaccard < 0.5:
            issues.append("palette_mismatch")

    axes_obs = inferred["axes"]
    if spec.axes.x_label and not axes_obs["x_label_present"]:
        issues.append("missing_x_label")
    if spec.axes.y_label and not axes_obs["y_label_present"]:
        issues.append("missing_y_label")

    if spec.axes.x_ticks is not None:
        diff = abs(spec.axes.x_ticks - axes_obs["x_tick_count_est"])
        deltas["x_tick_delta"] = diff
        if diff > 1:
            issues.append("x_tick_count_off")
    if spec.axes.y_ticks is not None:
        diff = abs(spec.axes.y_ticks - axes_obs["y_tick_count_est"])
        deltas["y_tick_delta"] = diff
        if diff > 1:
            issues.append("y_tick_count_off")

    penalty = min(len(issues) * 0.12, 1.0)
    score = max(0.0, 1.0 - penalty)

    return {"score": score, "issues": issues, "deltas": deltas}


def build_vlm_prompt(spec: ExpectedSpec, obs: dict, issues: List[str]) -> str:
    spec_summary = json.dumps(spec.__dict__, indent=2, default=lambda o: o.__dict__)
    obs_summary = json.dumps(
        {
            "stats": obs["stats"],
            "inferred": obs["inferred"],
            "palette": _palette_from_observation(obs["palette"])[:5],
        },
        indent=2,
    )
    issue_text = "- " + "\n- ".join(issues) if issues else "None detected"
    instructions = (
        "You are reviewing an LLM-generated chart image. "
        "Using only the textual spec and observed properties above, "
        "explain concrete fixes to the plotting code that would resolve the listed mismatches. "
        "Do not invent requirements beyond the provided spec."
    )
    return (
        "=== Expected Spec ===\n"
        f"{spec_summary}\n\n"
        "=== Observed Chart Properties ===\n"
        f"{obs_summary}\n\n"
        "=== Detected Issues ===\n"
        f"{issue_text}\n\n"
        f"{instructions}"
    )


def evaluate_generated_chart(spec_path: str, gen_img_path: str) -> dict:
    spec = from_json_file(spec_path)
    obs = observe_generated_chart(gen_img_path)
    cmp = compare_spec_to_observation(spec, obs)
    prompt = build_vlm_prompt(spec, obs, cmp["issues"])
    return {
        "spec": json.loads(json.dumps(spec, default=lambda o: o.__dict__)),
        "observation": obs,
        "comparison": cmp,
        "vlm_prompt": prompt,
    }


__all__ = [
    "observe_generated_chart",
    "compare_spec_to_observation",
    "build_vlm_prompt",
    "evaluate_generated_chart",
]
