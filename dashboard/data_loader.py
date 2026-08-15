"""Load saved experiment artifacts for the Streamlit dashboard."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _to_number(value: Any) -> Any:
    if value in ("", None):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _preview(value: Any, limit: int = 180) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def discover_runs(outputs_root: str | Path) -> list[dict[str, Any]]:
    """Find saved report runs under an outputs directory."""
    root = Path(outputs_root)
    runs = []
    for report_path in root.glob("**/report.json"):
        report = _read_json(report_path, {})
        run_dir = report_path.parent
        config = _read_json(run_dir / "config.json", {})
        run_id = report.get("run_id", run_dir.name)
        runs.append(
            {
                "run_id": run_id,
                "run_dir": run_dir,
                "report_path": report_path,
                "metrics": report.get("metrics", {}),
                "config": config,
                "label": _run_label(run_id, config),
                "sort_key": str(run_dir),
            }
        )
    return sorted(runs, key=lambda run: run["sort_key"], reverse=True)


def _run_label(run_id: str, config: dict[str, Any]) -> str:
    alpha = config.get("alpha")
    model = config.get("model_name")
    if alpha is not None and model:
        return f"{run_id} | {model} | alpha={alpha}"
    if alpha is not None:
        return f"{run_id} | alpha={alpha}"
    return run_id


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load all saved artifacts for a single run directory."""
    path = Path(run_dir)
    return {
        "run_dir": path,
        "report": _read_json(path / "report.json", {}),
        "config": _read_json(path / "config.json", {}),
        "prompt_csv_rows": _read_csv(path / "prompt_results.csv"),
        "failure_markdown": (path / "failure_analysis.md").read_text(encoding="utf-8")
        if (path / "failure_analysis.md").exists()
        else "",
        "summary_markdown": (path / "summary.md").read_text(encoding="utf-8")
        if (path / "summary.md").exists()
        else "",
    }


def load_alpha_sweep(outputs_root: str | Path) -> list[dict[str, Any]]:
    """Load alpha sweep rows from saved CSV artifacts."""
    rows = _read_csv(Path(outputs_root) / "alpha_sweep" / "results.csv")
    return [{key: _to_number(value) for key, value in row.items()} for row in rows]


def prompt_results_as_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten report prompt results into table-friendly rows."""
    rows = []
    for result in report.get("prompt_results", []):
        failure_categories = result.get("failure_categories", [])
        rows.append(
            {
                "id": result.get("id", ""),
                "category": result.get("category", ""),
                "prompt": result.get("prompt", ""),
                "baseline_completion": result.get("baseline_completion", ""),
                "unlearned_completion": result.get("unlearned_completion", ""),
                "baseline_target_probability": result.get("baseline_target_probability", 0.0),
                "unlearned_target_probability": result.get("unlearned_target_probability", 0.0),
                "unlearned_generic_probability": result.get("unlearned_generic_probability", 0.0),
                "target_unlearned_delta": result.get("target_unlearned_delta", 0.0),
                "generic_unlearned_delta": result.get("generic_unlearned_delta", 0.0),
                "failure_labels": ", ".join(failure_categories),
            }
        )
    return rows


def failure_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return failed prompt rows with compact completion previews."""
    rows = []
    for result in report.get("prompt_results", []):
        failure_categories = result.get("failure_categories", [])
        if not failure_categories:
            continue
        rows.append(
            {
                "id": result.get("id", ""),
                "category": result.get("category", ""),
                "prompt": result.get("prompt", ""),
                "failure_labels": ", ".join(failure_categories),
                "baseline_preview": _preview(result.get("baseline_completion", "")),
                "unlearned_preview": _preview(result.get("unlearned_completion", "")),
            }
        )
    return rows
