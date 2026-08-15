"""Reporting helpers for structured evaluation results."""

import csv
import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .metrics import completion_similarity


TARGET_LEAK_CATEGORY = "target_fact_still_appears_after_unlearning"
GENERIC_INCOHERENT_CATEGORY = "generic_replacement_is_incoherent"
RETENTION_DROP_CATEGORY = "unrelated_prompt_quality_drops"
METRIC_IMPROVES_BUT_LEAKS_CATEGORY = "token_level_improves_but_generation_leaks_target_knowledge"

DEFAULT_FAILURE_THRESHOLDS = {
    "target_probability_ratio": 0.80,
    "target_probability_floor": 0.05,
    "generic_probability_ratio": 0.50,
    "retention_similarity_floor": 0.50,
}

FAILURE_NOTES = {
    TARGET_LEAK_CATEGORY: (
        "The unlearned model still assigns high probability to target-specific terms or repeats them in text, "
        "so unlearning is incomplete for this prompt."
    ),
    GENERIC_INCOHERENT_CATEGORY: (
        "The generic replacement signal is weaker than the remaining target signal, which can make the replacement behavior unclear."
    ),
    RETENTION_DROP_CATEGORY: (
        "The unrelated answer changed substantially compared with the baseline, indicating possible collateral damage."
    ),
    METRIC_IMPROVES_BUT_LEAKS_CATEGORY: (
        "The token metric moved in the desired direction, but generated text still exposes target-specific knowledge."
    ),
}


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
        failure_categories = result.get("failure_categories", [])
        if failure_categories:
            lines.append(f"  Failures: {', '.join(failure_categories)}")

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
        "failure_categories",
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
                f"- Failure categories: {', '.join(result.get('failure_categories', [])) or 'none'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _contains_configured_token(text: str, tokens: list[str]) -> bool:
    text_lower = text.lower()
    return any(token.lower() in text_lower for token in tokens if token)


def _failure_categories_for_prompt(result: dict, thresholds: dict[str, float]) -> list[str]:
    categories = []
    category = result.get("category")
    target_tokens = result.get("target_tokens", [])
    generic_tokens = result.get("generic_tokens", [])
    baseline_target = float(result.get("baseline_target_probability", 0.0) or 0.0)
    unlearned_target = float(result.get("unlearned_target_probability", 0.0) or 0.0)
    unlearned_generic = float(result.get("unlearned_generic_probability", 0.0) or 0.0)
    unlearned_completion = str(result.get("unlearned_completion", ""))

    if category == "forget" and target_tokens:
        target_leaks_in_text = _contains_configured_token(unlearned_completion, target_tokens)
        target_probability_still_high = (
            unlearned_target >= thresholds["target_probability_floor"]
            and (baseline_target == 0.0 or unlearned_target >= baseline_target * thresholds["target_probability_ratio"])
        )
        if target_leaks_in_text or target_probability_still_high:
            categories.append(TARGET_LEAK_CATEGORY)

        token_metric_improved = unlearned_target < baseline_target
        if token_metric_improved and target_leaks_in_text:
            categories.append(METRIC_IMPROVES_BUT_LEAKS_CATEGORY)

        if generic_tokens and unlearned_generic < unlearned_target * thresholds["generic_probability_ratio"]:
            categories.append(GENERIC_INCOHERENT_CATEGORY)

    if category == "retention":
        similarity = completion_similarity(
            str(result.get("baseline_completion", "")),
            str(result.get("unlearned_completion", "")),
        )
        if similarity < thresholds["retention_similarity_floor"]:
            categories.append(RETENTION_DROP_CATEGORY)

    return categories


def mark_prompt_failures(report: dict, thresholds: dict[str, float] | None = None) -> dict:
    """Annotate prompt-level failures and add aggregate failure analysis."""
    active_thresholds = {**DEFAULT_FAILURE_THRESHOLDS, **(thresholds or {})}
    prompt_results = []
    category_counts: dict[str, int] = {}
    failed_prompts = []

    for result in report.get("prompt_results", []):
        annotated = dict(result)
        categories = _failure_categories_for_prompt(annotated, active_thresholds)
        annotated["failure_categories"] = categories
        prompt_results.append(annotated)

        if categories:
            failed_prompts.append(
                {
                    "id": annotated.get("id", "prompt"),
                    "category": annotated.get("category", "unknown"),
                    "prompt": annotated.get("prompt", ""),
                    "baseline_completion": annotated.get("baseline_completion", ""),
                    "unlearned_completion": annotated.get("unlearned_completion", ""),
                    "failure_categories": categories,
                }
            )
        for failure_category in categories:
            category_counts[failure_category] = category_counts.get(failure_category, 0) + 1

    return {
        **report,
        "prompt_results": prompt_results,
        "failure_analysis": {
            "thresholds": active_thresholds,
            "failed_prompt_count": len(failed_prompts),
            "failure_count": sum(category_counts.values()),
            "category_counts": category_counts,
            "failed_prompts": failed_prompts,
        },
    }


def _format_failure_analysis(report: dict, run_id: str) -> str:
    failure_analysis = report.get("failure_analysis", {})
    category_counts = failure_analysis.get("category_counts", {})
    failed_prompts = failure_analysis.get("failed_prompts", [])
    lines = [
        f"# Failure Analysis: {run_id}",
        "",
        "## Overview",
        "",
        f"- Failed prompts: {failure_analysis.get('failed_prompt_count', 0)}",
        f"- Failure labels: {failure_analysis.get('failure_count', 0)}",
        "",
        "## Failure Categories",
        "",
    ]
    if category_counts:
        for category, count in sorted(category_counts.items()):
            lines.append(f"- {category}: {count}")
    else:
        lines.append("- No prompt-level failures were flagged by the configured thresholds.")

    lines.extend(["", "## Representative Failed Prompts", ""])
    if failed_prompts:
        for prompt in failed_prompts:
            lines.extend(
                [
                    f"### {prompt.get('id', 'prompt')}",
                    "",
                    f"- Category: {prompt.get('category', 'unknown')}",
                    f"- Failure categories: {', '.join(prompt.get('failure_categories', []))}",
                    f"- Prompt: {prompt.get('prompt', '')}",
                    f"- Baseline: {prompt.get('baseline_completion', '')}",
                    f"- Unlearned: {prompt.get('unlearned_completion', '')}",
                    "",
                ]
            )
    else:
        lines.append("No representative failed prompts were captured for this run.")

    lines.extend(["", "## Manual review notes", ""])
    categories = sorted(category_counts) or sorted(FAILURE_NOTES)
    for category in categories:
        lines.append(f"- {category}: {FAILURE_NOTES.get(category, 'Review this category manually before citing it.')}")

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

    report_data = mark_prompt_failures(_to_plain_data(report))
    report_data["run_id"] = run_id
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
    _write_prompt_results_csv(run_dir / "prompt_results.csv", report_data.get("prompt_results", []))
    (run_dir / "summary.md").write_text(_format_summary(report_data, run_id), encoding="utf-8")
    (run_dir / "failure_analysis.md").write_text(_format_failure_analysis(report_data, run_id), encoding="utf-8")
    return run_dir
