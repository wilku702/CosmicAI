from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chart_split.eval_generated import compare_spec_to_observation, observe_generated_chart
from chart_split.spec_schema import AxisSpec, ExpectedSpec, LegendSpec, SeriesSpec


def _draw_axes(canvas: np.ndarray) -> None:
    h, w = canvas.shape[:2]
    cv2.line(canvas, (60, h - 60), (w - 40, h - 60), (0, 0, 0), 2)
    cv2.line(canvas, (60, h - 60), (60, 40), (0, 0, 0), 2)
    for offset in range(1, 6):
        x = 60 + offset * 80
        cv2.line(canvas, (x, h - 65), (x, h - 55), (0, 0, 0), 2)
    for offset in range(1, 5):
        y = h - 60 - offset * 70
        cv2.line(canvas, (55, y), (65, y), (0, 0, 0), 2)


def _create_bar_chart(path: Path) -> None:
    canvas = np.full((480, 640, 3), 255, dtype=np.uint8)
    _draw_axes(canvas)
    colors = [(60, 120, 220), (60, 160, 100), (180, 80, 80)]
    for idx, color in enumerate(colors):
        for bar in range(3):
            x0 = 90 + bar * 120 + idx * 20
            x1 = x0 + 18
            height = 80 + (idx + bar) * 20
            cv2.rectangle(canvas, (x0, 360), (x1, 360 - height), color, -1)
    # legend top-right
    legend_x = 470
    legend_y = 60
    for idx, color in enumerate(colors):
        cv2.rectangle(canvas, (legend_x, legend_y + idx * 30), (legend_x + 25, legend_y + 20 + idx * 30), color, -1)
        cv2.rectangle(canvas, (legend_x, legend_y + idx * 30), (legend_x + 90, legend_y + 20 + idx * 30), (0, 0, 0), 1)
    cv2.imwrite(str(path), canvas)


def _create_line_chart(path: Path) -> None:
    canvas = np.full((480, 640, 3), 255, dtype=np.uint8)
    _draw_axes(canvas)
    pts = np.array([[80, 360], [180, 320], [260, 300], [360, 250], [460, 220], [560, 200]], dtype=np.int32)
    cv2.polylines(canvas, [pts], False, (50, 50, 200), 3)
    cv2.imwrite(str(path), canvas)


def test_observe_generated_chart_bar(tmp_path: Path) -> None:
    image_path = tmp_path / "bar.png"
    _create_bar_chart(image_path)

    observation = observe_generated_chart(str(image_path))
    inferred = observation["inferred"]

    assert inferred["chart_type"] == "bar"
    assert inferred["legend_present"] is True
    assert inferred["series_count"] >= 3
    assert inferred["axes"]["x_tick_count_est"] >= 3


def test_compare_spec_flags_mismatches(tmp_path: Path) -> None:
    image_path = tmp_path / "line.png"
    _create_line_chart(image_path)
    observation = observe_generated_chart(str(image_path))

    spec = ExpectedSpec(
        chart_type="bar",
        legend=LegendSpec(present=True, position="top-right"),
        series=SeriesSpec(count=3),
        axes=AxisSpec(x_label="Time", y_label="Value", x_ticks=5, y_ticks=4),
    )

    comparison = compare_spec_to_observation(spec, observation)

    assert comparison["score"] < 1.0
    assert "wrong_chart_type" in comparison["issues"]
    assert "missing_legend" in comparison["issues"]
    assert "missing_x_label" in comparison["issues"]
    assert "missing_y_label" in comparison["issues"]
