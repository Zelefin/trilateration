"""Command-line interface for trilateration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from trilateration.solver import Anchor, TrilaterationError, solve_position


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trilat")
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve_parser = subparsers.add_parser("solve", help="solve target GPS position from three anchors")
    solve_parser.add_argument("input", type=Path, help="JSON or CSV file with anchor measurements")
    solve_parser.add_argument("--target-alt", type=float, help="target altitude in meters")
    solve_parser.add_argument("--input-format", choices=["auto", "json", "csv"], default="auto")
    solve_parser.add_argument("--max-rms-residual", type=float, default=None)
    solve_parser.add_argument("--pretty", action="store_true", help="pretty-print JSON output")

    plot_parser = subparsers.add_parser("plot", help="write a PNG plot of anchors, range circles, and solution")
    plot_parser.add_argument("input", type=Path, help="JSON or CSV file with anchor measurements")
    plot_parser.add_argument("--target-alt", type=float, help="target altitude in meters")
    plot_parser.add_argument("--input-format", choices=["auto", "json", "csv"], default="auto")
    plot_parser.add_argument("--max-rms-residual", type=float, default=None)
    plot_parser.add_argument("--output", type=Path, default=Path("trilateration_plot.png"))
    plot_parser.add_argument("--pretty", action="store_true", help="pretty-print JSON output")

    args = parser.parse_args(argv)
    if args.command == "solve":
        return _solve_command(args)
    if args.command == "plot":
        return _plot_command(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _solve_command(args: argparse.Namespace) -> int:
    try:
        anchors, file_target_alt = _load_anchors(args.input, args.input_format)
        target_alt = args.target_alt if args.target_alt is not None else file_target_alt
        if target_alt is None:
            raise TrilaterationError("Provide --target-alt or target_alt_m in the JSON input")

        result = solve_position(anchors, target_alt_m=target_alt, max_rms_residual_m=args.max_rms_residual)
        print(json.dumps(result.to_dict(), indent=2 if args.pretty else None, sort_keys=True))
        return 0
    except (OSError, KeyError, ValueError, TrilaterationError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 1


def _plot_command(args: argparse.Namespace) -> int:
    try:
        from trilateration.visualization import plot_solution

        anchors, file_target_alt = _load_anchors(args.input, args.input_format)
        target_alt = args.target_alt if args.target_alt is not None else file_target_alt
        if target_alt is None:
            raise TrilaterationError("Provide --target-alt or target_alt_m in the JSON input")

        result = solve_position(anchors, target_alt_m=target_alt, max_rms_residual_m=args.max_rms_residual)
        plot_solution(anchors, result, args.output)
        payload = {"output": str(args.output), "solution": result.to_dict()}
        print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
        return 0
    except (OSError, KeyError, RuntimeError, ValueError, TrilaterationError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 1


def _load_anchors(path: Path, input_format: str) -> tuple[list[Anchor], float | None]:
    resolved_format = input_format
    if resolved_format == "auto":
        suffix = path.suffix.lower()
        if suffix == ".json":
            resolved_format = "json"
        elif suffix == ".csv":
            resolved_format = "csv"
        else:
            raise ValueError("Cannot infer input format; use --input-format json or --input-format csv")

    if resolved_format == "json":
        return _load_json(path)
    if resolved_format == "csv":
        return _load_csv(path), None
    raise ValueError(f"Unsupported input format: {resolved_format}")


def _load_json(path: Path) -> tuple[list[Anchor], float | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    target_alt: float | None = None
    if isinstance(data, dict):
        raw_anchors = data["anchors"]
        if data.get("target_alt_m") is not None:
            target_alt = float(data["target_alt_m"])
    elif isinstance(data, list):
        raw_anchors = data
    else:
        raise ValueError("JSON input must be a list of anchors or an object with anchors")

    if not isinstance(raw_anchors, list):
        raise ValueError("JSON anchors must be a list")
    return [_anchor_from_any(item) for item in raw_anchors], target_alt


def _load_csv(path: Path) -> list[Anchor]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [_anchor_from_any(row) for row in csv.DictReader(handle)]


def _anchor_from_any(item: Any) -> Anchor:
    if not isinstance(item, dict):
        raise ValueError("Each anchor must be an object")
    return Anchor.from_mapping(item)


if __name__ == "__main__":
    raise SystemExit(main())
