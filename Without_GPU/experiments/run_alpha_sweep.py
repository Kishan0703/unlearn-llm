"""Run and summarize alpha trade-off experiments."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Callable, Iterable

from Without_GPU.unlearn import UnlearnConfig, unlearn
from Without_GPU.unlearn.constants import SYNTHETIC_UNIVERSE_DIR


DEFAULT_ALPHA_VALUES = [0.0, 2.0, 5.0, 10.0]
RESULT_FIELDS = [
    "alpha",
    "forgetting_score",
    "generic_replacement_score",
    "retention_score",
    "prompt_count",
    "run_id",
    "report_path",
]


def _format_alpha(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p")


def _as_float(value) -> float:
    if value is None:
        return 0.0
    return float(value)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _write_png(path: Path, width: int, height: int, pixels: list[list[tuple[int, int, int]]]) -> None:
    raw_rows = []
    for row in pixels:
        raw_rows.append(b"\x00" + b"".join(bytes(pixel) for pixel in row))
    payload = zlib.compress(b"".join(raw_rows), level=9)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", payload)
        + _png_chunk(b"IEND", b"")
    )


def _draw_line(pixels, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    width = len(pixels[0])
    height = len(pixels)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            pixels[y0][x0] = color
        if x0 == x1 and y0 == y1:
            break
        double_error = 2 * error
        if double_error >= dy:
            error += dy
            x0 += sx
        if double_error <= dx:
            error += dx
            y0 += sy


def _draw_square(pixels, x: int, y: int, color: tuple[int, int, int], radius: int = 3) -> None:
    width = len(pixels[0])
    height = len(pixels)
    for yy in range(max(0, y - radius), min(height, y + radius + 1)):
        for xx in range(max(0, x - radius), min(width, x + radius + 1)):
            pixels[yy][xx] = color


def write_tradeoff_chart(rows: list[dict], path: Path) -> None:
    """Write a small PNG chart comparing forgetting and retention by alpha."""
    width, height = 760, 420
    margin_left, margin_right, margin_top, margin_bottom = 70, 35, 35, 65
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    pixels = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]

    axis_color = (45, 45, 45)
    grid_color = (225, 225, 225)
    forgetting_color = (41, 98, 181)
    retention_color = (42, 132, 72)

    x_axis_y = margin_top + plot_height
    y_axis_x = margin_left
    _draw_line(pixels, y_axis_x, margin_top, y_axis_x, x_axis_y, axis_color)
    _draw_line(pixels, y_axis_x, x_axis_y, width - margin_right, x_axis_y, axis_color)

    for tick in range(6):
        y = margin_top + round(plot_height * tick / 5)
        _draw_line(pixels, y_axis_x, y, width - margin_right, y, grid_color)

    if not rows:
        _write_png(path, width, height, pixels)
        return

    alphas = [_as_float(row["alpha"]) for row in rows]
    min_alpha = min(alphas)
    max_alpha = max(alphas)
    alpha_span = max(max_alpha - min_alpha, 1.0)

    def point(alpha: float, value: float) -> tuple[int, int]:
        x = margin_left + round(((alpha - min_alpha) / alpha_span) * plot_width)
        clipped = max(0.0, min(1.0, value))
        y = margin_top + round((1.0 - clipped) * plot_height)
        return x, y

    series = [
        ("forgetting_score", forgetting_color),
        ("retention_score", retention_color),
    ]
    for key, color in series:
        previous = None
        for row in rows:
            current = point(_as_float(row["alpha"]), _as_float(row.get(key)))
            if previous is not None:
                _draw_line(pixels, previous[0], previous[1], current[0], current[1], color)
            _draw_square(pixels, current[0], current[1], color)
            previous = current

    path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(path, width, height, pixels)


def identify_best_tradeoff(rows: Iterable[dict]) -> dict:
    """Choose the alpha with low forgetting score while retaining unrelated behavior."""
    candidates = list(rows)
    if not candidates:
        return {}

    def score(row: dict) -> float:
        retention = _as_float(row.get("retention_score"))
        forgetting_reduction = 1.0 - _as_float(row.get("forgetting_score"))
        generic = _as_float(row.get("generic_replacement_score"))
        return (0.50 * retention) + (0.35 * forgetting_reduction) + (0.15 * generic)

    return max(candidates, key=score)


def _format_metric(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(number):
        return f"{number:.4f}"
    return str(value)


def _summary_markdown(rows: list[dict]) -> str:
    best = identify_best_tradeoff(rows)
    lines = [
        "# Alpha Sweep Summary",
        "",
        "This sweep compares how the unlearning strength alpha changes target forgetting and unrelated prompt retention.",
        "",
    ]
    if best:
        lines.extend(
            [
                f"Best trade-off alpha: {best['alpha']}",
                "",
                "- The best trade-off is selected by rewarding retention, lower target familiarity, and generic replacement probability.",
                f"- Forgetting score at best alpha: {_format_metric(best.get('forgetting_score'))}",
                f"- Retention score at best alpha: {_format_metric(best.get('retention_score'))}",
                f"- Generic replacement score at best alpha: {_format_metric(best.get('generic_replacement_score'))}",
                "",
            ]
        )

    lines.extend(["## Per-Alpha Results", ""])
    for row in rows:
        lines.append(
            "- alpha {alpha}: forgetting={forgetting}, retention={retention}, generic={generic}".format(
                alpha=row["alpha"],
                forgetting=_format_metric(row.get("forgetting_score")),
                retention=_format_metric(row.get("retention_score")),
                generic=_format_metric(row.get("generic_replacement_score")),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def save_alpha_sweep_outputs(rows: list[dict], output_dir: str | Path) -> Path:
    """Save aggregate alpha sweep artifacts and return the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: _as_float(row["alpha"]))

    with (output_path / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})

    write_tradeoff_chart(sorted_rows, output_path / "forgetting_vs_retention.png")
    (output_path / "summary.md").write_text(_summary_markdown(sorted_rows), encoding="utf-8")
    return output_path


def _report_row(alpha: float, report_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})
    return {
        "alpha": alpha,
        "forgetting_score": metrics.get("forgetting_score", 0.0),
        "generic_replacement_score": metrics.get("generic_replacement_score", 0.0),
        "retention_score": metrics.get("retention_score", 0.0),
        "prompt_count": metrics.get("prompt_count", 0),
        "run_id": report.get("run_id", report_path.parent.name),
        "report_path": str(report_path),
    }


def _find_latest_report(report_root: Path, alpha: float) -> Path:
    candidates = []
    for path in report_root.glob("*/report.json"):
        config_path = path.parent / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if float(config.get("alpha", -1.0)) == float(alpha):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No report.json found for alpha={alpha} under {report_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_alpha_config(base_config: UnlearnConfig, alpha: float, model_output_root: Path, report_root: Path) -> UnlearnConfig:
    """Copy the base config and point model/report paths at alpha-specific locations."""
    config = copy.deepcopy(base_config)
    alpha_slug = _format_alpha(alpha)
    config.alpha = alpha
    config.output_dir = str(model_output_root / f"alpha_{alpha_slug}")
    config.reinforced_model_dir = str(Path(config.output_dir) / "reinforced")
    config.unlearned_model_dir = str(Path(config.output_dir) / "unlearned")
    config.report_dir = str(report_root)
    config.run_name = f"alpha-sweep-alpha-{alpha_slug}"
    return config


def run_alpha_sweep(
    base_config: UnlearnConfig,
    alpha_values: Iterable[float],
    sweep_dir: str | Path = "outputs/alpha_sweep",
    pipeline_runner: Callable[[UnlearnConfig], object] = unlearn,
) -> list[dict]:
    """Run the pipeline for every alpha and save combined sweep artifacts."""
    sweep_path = Path(sweep_dir)
    report_root = sweep_path / "runs"
    model_output_root = Path(base_config.output_dir)
    rows = []
    for alpha in alpha_values:
        config = build_alpha_config(base_config, float(alpha), model_output_root, report_root)
        result = pipeline_runner(config)
        report_path = Path(result) if isinstance(result, (str, Path)) else _find_latest_report(report_root, config.alpha)
        rows.append(_report_row(config.alpha, report_path))

    save_alpha_sweep_outputs(rows, sweep_path)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run alpha trade-off analysis for the synthetic unlearning dataset.")
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHA_VALUES)
    parser.add_argument(
        "--target_text",
        default=str(SYNTHETIC_UNIVERSE_DIR / "target_corpus.txt"),
        help="Synthetic target corpus path.",
    )
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--reinforce_epochs", type=int, default=3)
    parser.add_argument("--unlearn_epochs", type=int, default=2)
    parser.add_argument("--reinforce_lr", type=float, default=3e-6)
    parser.add_argument("--unlearn_lr", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", default="output/alpha_sweep_models")
    parser.add_argument("--sweep_dir", default="outputs/alpha_sweep")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    base_config = UnlearnConfig(
        model_name=args.model_name,
        device=args.device,
        target_text_path=args.target_text,
        output_dir=args.output_dir,
        block_size=args.block_size,
        reinforce_epochs=args.reinforce_epochs,
        unlearn_epochs=args.unlearn_epochs,
        reinforce_lr=args.reinforce_lr,
        unlearn_lr=args.unlearn_lr,
        reinforce_batch_size=args.batch_size,
        unlearn_batch_size=args.batch_size,
    )
    rows = run_alpha_sweep(base_config, args.alphas, args.sweep_dir)
    best = identify_best_tradeoff(rows)
    print(f"Alpha sweep saved to: {args.sweep_dir}")
    if best:
        print(f"Best trade-off alpha: {best['alpha']}")


if __name__ == "__main__":
    main()
