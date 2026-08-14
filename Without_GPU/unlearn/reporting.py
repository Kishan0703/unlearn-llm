"""Human-readable reporting helpers for structured evaluation results."""


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
