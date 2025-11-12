from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

LegendPosition = Literal[
    "top",
    "top-left",
    "top-right",
    "bottom",
    "bottom-left",
    "bottom-right",
    "left",
    "right",
    "inside",
    "none",
]

ChartType = Literal["bar", "line", "scatter", "area", "hist", "box", "pie", "heatmap", "other"]


@dataclass
class AxisSpec:
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    x_ticks: Optional[int] = None
    y_ticks: Optional[int] = None


@dataclass
class LegendSpec:
    present: Optional[bool] = None
    position: Optional[LegendPosition] = None


@dataclass
class SeriesSpec:
    count: Optional[int] = None
    labels: Optional[List[str]] = None


@dataclass
class LayoutSpec:
    plot_area_ratio_min: Optional[float] = None


@dataclass
class ExpectedSpec:
    chart_type: ChartType
    axes: AxisSpec = field(default_factory=AxisSpec)
    legend: LegendSpec = field(default_factory=LegendSpec)
    series: SeriesSpec = field(default_factory=SeriesSpec)
    palette: Optional[List[str]] = None
    layout: LayoutSpec = field(default_factory=LayoutSpec)
    extras: Optional[Dict[str, object]] = None


def from_json_file(path: str | Path) -> ExpectedSpec:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    return ExpectedSpec(
        chart_type=payload.get("chart_type", "other"),
        axes=AxisSpec(**payload.get("axes", {})),
        legend=LegendSpec(**payload.get("legend", {})),
        series=SeriesSpec(**payload.get("series", {})),
        palette=payload.get("palette"),
        layout=LayoutSpec(**payload.get("layout", {})),
        extras=payload.get("extras"),
    )


__all__ = ["ExpectedSpec", "from_json_file"]
