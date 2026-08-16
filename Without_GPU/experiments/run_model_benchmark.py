"""Run and summarize multi-model unlearning experiments."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Callable, Iterable

from Without_GPU.unlearn import UnlearnConfig, unlearn
from Without_GPU.unlearn.constants import SYNTHETIC_UNIVERSE_DIR


DEFAULT_MODEL_NAMES = ["openai-community/gpt2-medium", "gpt2"]
RESULT_FIELDS = [
    "model_name",
    "forgetting_score",
    "generic_replacement_score",
    "retention_score",
    "prompt_count",
    "run_id",
    "report_path",
]


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "-").replace(" ", "-")


def _as_float(value) -> float:
    if value is None:
        return 0.0
    return float(value)


def _format_metric(value) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def build_model_config(base_config: UnlearnConfig, model_name: str, benchmark_dir: Path) -> UnlearnConfig:
    """Copy a base config and point paths at a model-specific benchmark run."""
    config = copy.deepcopy(base_config)
    slug = _model_slug(model_name)
    model_output_root = Path(base_config.output_dir)
    config.model_name = model_name
    config.output_dir = str(model_output_root / slug)
    config.reinforced_model_dir = str(Path(config.output_dir) / "reinforced")
    config.unlearned_model_dir = str(Path(config.output_dir) / "unlearned")
    config.report_dir = str(benchmark_dir / "runs")
    config.run_name = f"model-benchmark-{slug}"
    return config


def identify_best_model(rows: Iterable[dict]) -> dict:
    """Choose the model with the best retention, forgetting, and replacement trade-off."""
    candidates = list(rows)
    if not candidates:
        return {}

    def score(row: dict) -> float:
        retention = _as_float(row.get("retention_score"))
        forgetting_reduction = 1.0 - _as_float(row.get("forgetting_score"))
        generic = _as_float(row.get("generic_replacement_score"))
        return (0.45 * retention) + (0.35 * forgetting_reduction) + (0.20 * generic)

    return max(candidates, key=score)


def _summary_markdown(rows: list[dict]) -> str:
    best = identify_best_model(rows)
    lines = [
        "# Model Benchmark Summary",
        "",
        "This benchmark compares model choices using the same synthetic target corpus, prompts, and report schema.",
        "",
    ]
    if best:
        lines.extend(
            [
                f"Best trade-off model: {best['model_name']}",
                "",
                "- The best model is selected by rewarding retention, lower target familiarity, and generic replacement probability.",
                f"- Forgetting score: {_format_metric(best.get('forgetting_score'))}",
                f"- Retention score: {_format_metric(best.get('retention_score'))}",
                f"- Generic replacement score: {_format_metric(best.get('generic_replacement_score'))}",
                "",
            ]
        )

    lines.extend(["## Per-Model Results", ""])
    for row in sorted(rows, key=lambda item: str(item["model_name"])):
        lines.append(
            "- {model}: forgetting={forgetting}, retention={retention}, generic={generic}".format(
                model=row["model_name"],
                forgetting=_format_metric(row.get("forgetting_score")),
                retention=_format_metric(row.get("retention_score")),
                generic=_format_metric(row.get("generic_replacement_score")),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def save_model_benchmark_outputs(rows: list[dict], output_dir: str | Path) -> Path:
    """Save model benchmark aggregate artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: str(row["model_name"]))

    with (output_path / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})

    (output_path / "summary.md").write_text(_summary_markdown(sorted_rows), encoding="utf-8")
    return output_path


def _report_row(model_name: str, report_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})
    return {
        "model_name": model_name,
        "forgetting_score": metrics.get("forgetting_score", 0.0),
        "generic_replacement_score": metrics.get("generic_replacement_score", 0.0),
        "retention_score": metrics.get("retention_score", 0.0),
        "prompt_count": metrics.get("prompt_count", 0),
        "run_id": report.get("run_id", report_path.parent.name),
        "report_path": str(report_path),
    }


def _find_latest_report(report_root: Path, model_name: str) -> Path:
    candidates = []
    for path in report_root.glob("*/report.json"):
        config_path = path.parent / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("model_name") == model_name:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No report.json found for model={model_name} under {report_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_model_benchmark(
    base_config: UnlearnConfig,
    model_names: Iterable[str],
    benchmark_dir: str | Path = "outputs/model_benchmark",
    pipeline_runner: Callable[[UnlearnConfig], object] = unlearn,
) -> list[dict]:
    """Run the unlearning pipeline for each model and save combined benchmark artifacts."""
    benchmark_path = Path(benchmark_dir)
    rows = []
    for model_name in model_names:
        config = build_model_config(base_config, model_name, benchmark_path)
        result = pipeline_runner(config)
        report_path = (
            Path(result)
            if isinstance(result, (str, Path))
            else _find_latest_report(Path(config.report_dir), config.model_name)
        )
        rows.append(_report_row(config.model_name, report_path))

    save_model_benchmark_outputs(rows, benchmark_path)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run multi-model comparison for the synthetic unlearning dataset.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_NAMES)
    parser.add_argument(
        "--target_text",
        default=str(SYNTHETIC_UNIVERSE_DIR / "target_corpus.txt"),
        help="Synthetic target corpus path.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--reinforce_epochs", type=int, default=3)
    parser.add_argument("--unlearn_epochs", type=int, default=2)
    parser.add_argument("--reinforce_lr", type=float, default=3e-6)
    parser.add_argument("--unlearn_lr", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", default="output/model_benchmark_models")
    parser.add_argument("--benchmark_dir", default="outputs/model_benchmark")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    base_config = UnlearnConfig(
        model_name=args.models[0],
        device=args.device,
        target_text_path=args.target_text,
        output_dir=args.output_dir,
        block_size=args.block_size,
        alpha=args.alpha,
        reinforce_epochs=args.reinforce_epochs,
        unlearn_epochs=args.unlearn_epochs,
        reinforce_lr=args.reinforce_lr,
        unlearn_lr=args.unlearn_lr,
        reinforce_batch_size=args.batch_size,
        unlearn_batch_size=args.batch_size,
    )
    rows = run_model_benchmark(base_config, args.models, args.benchmark_dir)
    best = identify_best_model(rows)
    print(f"Model benchmark saved to: {args.benchmark_dir}")
    if best:
        print(f"Best trade-off model: {best['model_name']}")


if __name__ == "__main__":
    main()
