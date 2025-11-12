from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2

from .eval_generated import (
    build_vlm_prompt,
    compare_spec_to_observation,
    observe_generated_chart,
)
from .guard import assert_no_gt_refs
from .preprocess import preprocess_for_llm, save_overlays
from .spec_schema import ExpectedSpec, from_json_file

SUPPORTED_SUFFIXES: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _collect_images(root: Path) -> List[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_SUFFIXES else []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _write_metadata(target: Path, artifacts: dict) -> None:
    payload = {
        "stats": artifacts["stats"],
        "component_stats": artifacts["component_stats"],
        "palette": artifacts["palette"],
        "region_descriptors": artifacts["region_descriptors"],
    }
    with target.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def process_directory(
    image_dir: str,
    output_dir: str,
    *,
    limit: int | None = None,
    max_dim: int = 768,
    pad_to_square: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_grid: Sequence[int] = (8, 8),
    adaptive_block_size: int = 35,
    adaptive_c: int = 7,
    skeleton_kernel: int = 3,
    palette_size: int = 5,
    verbose: bool = True,
) -> None:
    root = Path(image_dir).expanduser()
    images = _collect_images(root)
    if limit is not None and limit > 0:
        images = images[:limit]
    if not images:
        raise FileNotFoundError(f"No supported images found in {image_dir}")

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    for idx, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            if verbose:
                print(f"[{idx}/{len(images)}] Skipping {image_path} (failed to load).")
            continue

        artifacts = preprocess_for_llm(
            image,
            max_dim=max_dim,
            pad_to_square=pad_to_square,
            clahe_clip_limit=clahe_clip_limit,
            clahe_grid=tuple(clahe_grid),
            adaptive_block_size=adaptive_block_size,
            adaptive_c=adaptive_c,
            skeleton_kernel=skeleton_kernel,
            palette_size=palette_size,
        )

        target = output / image_path.stem
        save_overlays(artifacts, target)
        _write_metadata(target / "observation.json", artifacts)

        if verbose:
            stats = artifacts["stats"]
            print(
                f"[{idx}/{len(images)}] {image_path.name} "
                f"(edge_density={stats['edge_density']:.3f})"
            )


def _run_eval_generated(args: argparse.Namespace) -> None:
    assert_no_gt_refs(args.spec)
    assert_no_gt_refs(args.image)
    spec = from_json_file(args.spec)

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Unable to load generated chart: {args.image}")

    artifacts = preprocess_for_llm(image)
    observation = observe_generated_chart(args.image, artifacts=artifacts)
    comparison = compare_spec_to_observation(spec, observation)
    prompt = build_vlm_prompt(spec, observation, comparison["issues"])

    out_dir = Path(args.out).expanduser()
    overlays_dir = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_overlays(artifacts, overlays_dir)

    def _write_json(path: Path, payload: object) -> None:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    spec_dict = json.loads(json.dumps(spec, default=lambda o: o.__dict__))
    _write_json(out_dir / "spec.json", spec_dict)
    _write_json(out_dir / "observation.json", observation)
    _write_json(out_dir / "comparison.json", comparison)
    (out_dir / "vlm_prompt.txt").write_text(prompt, encoding="utf-8")


def _run_image_tests(args: argparse.Namespace) -> None:
    image_dir = Path(args.images).expanduser()
    images = _collect_images(image_dir)
    if args.limit is not None and args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise FileNotFoundError(f"No supported images found in {args.images}")

    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    summary: List[dict] = []
    for idx, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            if not args.quiet:
                print(f"[{idx}/{len(images)}] Skipping {image_path} (failed to load).")
            continue

        artifacts = preprocess_for_llm(
            image,
            max_dim=args.max_dim,
            pad_to_square=args.pad_to_square,
            clahe_clip_limit=args.clahe_clip,
            clahe_grid=tuple(args.clahe_grid),
            adaptive_block_size=args.adaptive_block,
            adaptive_c=args.adaptive_c,
            skeleton_kernel=args.skeleton_kernel,
            palette_size=args.palette_size,
        )
        observation = observe_generated_chart(str(image_path), artifacts=artifacts)

        target = out_root / image_path.stem
        save_overlays(artifacts, target / "overlays")
        with (target / "observation.json").open("w", encoding="utf-8") as fh:
            json.dump(observation, fh, indent=2)

        summary.append(
            {
                "image": str(image_path),
                "inferred": observation["inferred"],
                "stats": observation["stats"],
            }
        )

        if not args.quiet:
            inferred = observation["inferred"]
            print(
                f"[{idx}/{len(images)}] {image_path.name} -> "
                f"type={inferred['chart_type']} legend={inferred['legend_present']} series={inferred['series_count']}"
            )

    with (out_root / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chart preprocessing and spec-only evaluation utilities."
    )
    subparsers = parser.add_subparsers(dest="command")

    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess images.")
    preprocess_parser.add_argument("image_dir", help="Directory containing chart images.")
    preprocess_parser.add_argument(
        "-o",
        "--output",
        default="preprocess_outputs",
        help="Directory where overlays/metadata are written.",
    )
    preprocess_parser.add_argument("--limit", type=int, default=None)
    preprocess_parser.add_argument("--max-dim", type=int, default=768)
    preprocess_parser.add_argument("--no-pad", action="store_false", dest="pad_to_square")
    preprocess_parser.add_argument("--clahe-clip", type=float, default=2.0)
    preprocess_parser.add_argument(
        "--clahe-grid",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=(8, 8),
    )
    preprocess_parser.add_argument("--adaptive-block", type=int, default=35)
    preprocess_parser.add_argument("--adaptive-c", type=int, default=7)
    preprocess_parser.add_argument("--skeleton-kernel", type=int, default=3)
    preprocess_parser.add_argument("--palette-size", type=int, default=5)
    preprocess_parser.add_argument("--quiet", action="store_true")

    eval_parser = subparsers.add_parser(
        "eval-gen",
        help="Evaluate a generated chart image against a textual spec.",
    )
    eval_parser.add_argument("--spec", required=True, help="Path to expected spec JSON.")
    eval_parser.add_argument("--image", required=True, help="Generated chart image.")
    eval_parser.add_argument(
        "--out",
        required=True,
        help="Directory where evaluation artifacts are written.",
    )

    test_parser = subparsers.add_parser(
        "test-images",
        help="Run observation diagnostics across an images folder.",
    )
    test_parser.add_argument("--images", required=True, help="Directory with generated images.")
    test_parser.add_argument("--out", required=True, help="Directory for test reports/overlays.")
    test_parser.add_argument("--limit", type=int, default=None)
    test_parser.add_argument("--max-dim", type=int, default=768)
    test_parser.add_argument("--no-pad", action="store_false", dest="pad_to_square")
    test_parser.add_argument("--clahe-clip", type=float, default=2.0)
    test_parser.add_argument(
        "--clahe-grid",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=(8, 8),
    )
    test_parser.add_argument("--adaptive-block", type=int, default=35)
    test_parser.add_argument("--adaptive-c", type=int, default=7)
    test_parser.add_argument("--skeleton-kernel", type=int, default=3)
    test_parser.add_argument("--palette-size", type=int, default=5)
    test_parser.add_argument("--quiet", action="store_true")

    return parser


def _inject_default_command(argv: List[str]) -> List[str]:
    if not argv:
        return ["preprocess"]
    first = argv[0]
    if first in {"preprocess", "eval-gen"}:
        return argv
    return ["preprocess", *argv]


def main(argv: List[str] | None = None) -> None:
    args_list = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    parsed = parser.parse_args(_inject_default_command(args_list))

    if parsed.command == "preprocess":
        process_directory(
            image_dir=parsed.image_dir,
            output_dir=parsed.output,
            limit=parsed.limit,
            max_dim=parsed.max_dim,
            pad_to_square=parsed.pad_to_square,
            clahe_clip_limit=parsed.clahe_clip,
            clahe_grid=tuple(parsed.clahe_grid),
            adaptive_block_size=parsed.adaptive_block,
            adaptive_c=parsed.adaptive_c,
            skeleton_kernel=parsed.skeleton_kernel,
            palette_size=parsed.palette_size,
            verbose=not parsed.quiet,
        )
        return

    if parsed.command == "eval-gen":
        _run_eval_generated(parsed)
        return

    if parsed.command == "test-images":
        _run_image_tests(parsed)
        return

    parser.print_help()


__all__ = ["process_directory", "main"]
