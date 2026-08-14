"""Reporting helpers for structured evaluation results."""

import csv
import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def format_console_report(report: dict) -> str:
    """Format structured evaluation results for terminal output."""
    lines = [
        "",
        "=" * 80,
        "STRUCTURED EVALUATION",
        "=" * 80,
        "",
        "Aggregate metrics:",
    ]
    for key, value in report.get("metrics", {}).items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")

    for result in report.get("prompt_results", []):
        lines.extend(
            [
                "",
                f"Prompt [{result.get('category', 'unknown')}:{result.get('id', '')}]: {result.get('prompt', '')}",
                f"  Baseline:  {result.get('baseline_completion', '')}",
                f"  Unlearned: {result.get('unlearned_completion', '')}",
            ]
        )
        if "unlearned_target_probability" in result:
            lines.append(f"  Target probability after unlearning: {result['unlearned_target_probability']:.4f}")
        if "unlearned_generic_probability" in result:
            lines.append(f"  Generic probability after unlearning: {result['unlearned_generic_probability']:.4f}")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "run"


def _format_number(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p")


def build_run_id(config, run_name: str = "", timestamp: str | None = None) -> str:
    """Build a stable run identifier from time and key experiment settings."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    model_name = _slug(Path(config.model_name).name)
    parts = [
        timestamp,
        _slug(run_name) if run_name else "",
        model_name,
        f"alpha-{_format_number(config.alpha)}",
        f"block-{config.block_size}",
    ]
    return "_".join(part for part in parts if part)


def _to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_prompt_results_csv(path: Path, prompt_results: list[dict]) -> None:
    fieldnames = [
        "id",
        "category",
        "prompt",
        "baseline_completion",
        "unlearned_completion",
        "target_tokens",
        "generic_tokens",
        "baseline_token_probs",
        "reinforced_token_probs",
        "unlearned_token_probs",
        "unlearned_target_probability",
        "unlearned_generic_probability",
        "target_reinforced_delta",
        "target_unlearned_delta",
        "generic_unlearned_delta",
    ]
    extra_fieldnames = sorted(
        {
            key
            for result in prompt_results
            for key in result
            if key not in fieldnames
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extra_fieldnames])
        writer.writeheader()
        for result in prompt_results:
            writer.writerow({key: _csv_value(result.get(key, "")) for key in writer.fieldnames})


def _format_summary(report: dict, run_id: str) -> str:
    metrics = report.get("metrics", {})
    prompt_results = report.get("prompt_results", [])
    lines = [
        f"# Experiment Summary: {run_id}",
        "",
        "## Aggregate Metrics",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Prompt Results", ""])
    for result in prompt_results:
        lines.extend(
            [
                f"### {result.get('id', 'prompt')}",
                "",
                f"- Category: {result.get('category', 'unknown')}",
                f"- Prompt: {result.get('prompt', '')}",
                f"- Baseline: {result.get('baseline_completion', '')}",
                f"- Unlearned: {result.get('unlearned_completion', '')}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def save_experiment_report(
    report: dict,
    config,
    report_dir: str | Path,
    run_name: str = "",
    timestamp: str | None = None,
) -> Path:
    """Save reusable report artifacts and return the run directory."""
    run_id = build_run_id(config, run_name=run_name, timestamp=timestamp)
    run_dir = Path(report_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report_data = {**_to_plain_data(report), "run_id": run_id}
    config_data = _to_plain_data(config)
    config_data["run_id"] = run_id

    (run_dir / "report.json").write_text(
        json.dumps(report_data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(config_data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_prompt_results_csv(run_dir / "prompt_results.csv", report.get("prompt_results", []))
    (run_dir / "summary.md").write_text(_format_summary(report, run_id), encoding="utf-8")
    return run_dir
