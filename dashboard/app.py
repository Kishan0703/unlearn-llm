"""Streamlit report viewer for saved unlearning experiment artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from data_loader import (
    discover_runs,
    failure_rows,
    load_alpha_sweep,
    load_run,
    prompt_results_as_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
ALPHA_CHART_PATH = OUTPUTS_ROOT / "alpha_sweep" / "forgetting_vs_retention.png"


def metric_value(metrics: dict[str, Any], key: str, default: Any = 0) -> Any:
    return metrics.get(key, default)


def format_metric(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b0f14;
            --panel: #121922;
            --panel-2: #17212c;
            --ink: #eef3f8;
            --muted: #95a3b3;
            --line: rgba(148, 163, 184, 0.20);
            --blue: #66d9ef;
            --green: #80ed99;
            --amber: #f6bd60;
            --red: #ff6b6b;
        }
        .stApp {
            background:
                radial-gradient(circle at 15% 0%, rgba(102, 217, 239, 0.09), transparent 28rem),
                linear-gradient(180deg, #0b0f14 0%, #0f141b 100%);
            color: var(--ink);
        }
        [data-testid="stHeader"] { background: rgba(11, 15, 20, 0.72); }
        [data-testid="stSidebar"] {
            background: #0d1218;
            border-right: 1px solid var(--line);
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1420px;
        }
        h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
        h1 {
            font-size: 2.15rem;
            line-height: 1.1;
            margin-bottom: 0.25rem;
        }
        h2 { font-size: 1.35rem; margin-top: 1.4rem; }
        h3 { font-size: 1.02rem; }
        .subtle {
            color: var(--muted);
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(23, 33, 44, 0.98), rgba(18, 25, 34, 0.98));
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            min-height: 112px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.16);
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            color: var(--ink);
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.15;
        }
        .metric-note {
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: 0.45rem;
        }
        .panel {
            background: rgba(18, 25, 34, 0.92);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
        }
        .completion {
            background: #0b1118;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            min-height: 220px;
            white-space: pre-wrap;
            color: #dce6ef;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.5rem;
        }
        .badge {
            border: 1px solid rgba(246, 189, 96, 0.5);
            background: rgba(246, 189, 96, 0.10);
            color: #ffd99a;
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            font-size: 0.76rem;
        }
        .failure-callout {
            border-left: 3px solid var(--amber);
            background: rgba(246, 189, 96, 0.08);
            padding: 0.85rem 1rem;
            border-radius: 6px;
        }
        .stDataFrame {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: Any, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{format_metric(value)}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def select_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [run["label"] for run in runs]
    selected = st.sidebar.selectbox("Run", labels, index=0)
    return runs[labels.index(selected)]


def render_alpha_sweep(rows: list[dict[str, Any]]) -> None:
    st.subheader("Alpha Trade-Off")
    if rows:
        df = pd.DataFrame(rows)
        columns = [column for column in ["forgetting_score", "retention_score", "generic_replacement_score"] if column in df]
        if columns:
            chart_df = df.set_index("alpha")[columns]
            st.line_chart(chart_df, height=280)
        st.dataframe(
            df[[column for column in ["alpha", "forgetting_score", "generic_replacement_score", "retention_score"] if column in df]],
            width="stretch",
            hide_index=True,
        )
    elif ALPHA_CHART_PATH.exists():
        st.image(str(ALPHA_CHART_PATH), caption="Saved alpha sweep chart")
    else:
        st.warning("No alpha sweep artifact found under outputs/alpha_sweep.")


def render_failure_callout(rows: list[dict[str, Any]]) -> None:
    st.subheader("Representative Failure")
    if not rows:
        st.markdown('<div class="panel">No failures were flagged for this run.</div>', unsafe_allow_html=True)
        return
    first = rows[0]
    st.markdown(
        f"""
        <div class="failure-callout">
            <strong>{first['id']}</strong> · {first['category']}<br>
            <span style="color:#95a3b3">{first['prompt']}</span>
            <div class="badge-row">
                {"".join(f'<span class="badge">{label.strip()}</span>' for label in first['failure_labels'].split(",") if label.strip())}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prompt_inspector(prompt_rows: list[dict[str, Any]]) -> None:
    st.subheader("Prompt Inspector")
    if not prompt_rows:
        st.warning("No prompt results found in this run.")
        return

    labels = [f"{row['id']} · {row['category']}" for row in prompt_rows]
    selected_label = st.selectbox("Prompt", labels, index=0)
    row = prompt_rows[labels.index(selected_label)]

    meta_cols = st.columns(3)
    with meta_cols[0]:
        render_metric_card("Target After", row.get("unlearned_target_probability", 0.0), "Target familiarity after unlearning")
    with meta_cols[1]:
        render_metric_card("Generic After", row.get("unlearned_generic_probability", 0.0), "Generic replacement probability")
    with meta_cols[2]:
        render_metric_card("Target Delta", row.get("target_unlearned_delta", 0.0), "Change from baseline target probability")

    st.markdown(f"**Prompt:** {row['prompt']}")
    completion_cols = st.columns(2)
    with completion_cols[0]:
        st.markdown("**Baseline Completion**")
        st.markdown(f"<div class='completion'>{row['baseline_completion']}</div>", unsafe_allow_html=True)
    with completion_cols[1]:
        st.markdown("**Unlearned Completion**")
        st.markdown(f"<div class='completion'>{row['unlearned_completion']}</div>", unsafe_allow_html=True)

    if row.get("failure_labels"):
        labels_html = "".join(
            f'<span class="badge">{label.strip()}</span>' for label in row["failure_labels"].split(",") if label.strip()
        )
        st.markdown(f"<div class='badge-row'>{labels_html}</div>", unsafe_allow_html=True)


def render_failure_table(rows: list[dict[str, Any]]) -> None:
    st.subheader("Failure Examples")
    if not rows:
        st.info("No failure rows to display for this run.")
        return
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=320)


def main() -> None:
    st.set_page_config(
        page_title="Unlearn-LLM Report Dashboard",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    runs = discover_runs(OUTPUTS_ROOT)
    st.sidebar.title("Unlearn-LLM")
    st.sidebar.caption("Saved report viewer")
    st.sidebar.write(f"Reports root: `{OUTPUTS_ROOT.relative_to(PROJECT_ROOT)}`")

    if not runs:
        st.title("Unlearn-LLM Report Dashboard")
        st.warning("No saved reports found under outputs/. Run an experiment first or check in report artifacts.")
        return

    selected_run = select_run(runs)
    run_data = load_run(selected_run["run_dir"])
    report = run_data["report"]
    config = run_data["config"]
    metrics = report.get("metrics", {})
    failures = failure_rows(report)
    prompts = prompt_results_as_rows(report)
    alpha_rows = load_alpha_sweep(OUTPUTS_ROOT)
    failed_prompt_count = report.get("failure_analysis", {}).get("failed_prompt_count", len(failures))

    st.title("Unlearn-LLM Report Dashboard")
    st.markdown(
        f"""
        <div class="subtle">
            Run <strong>{report.get('run_id', selected_run['run_id'])}</strong>
            · model <strong>{config.get('model_name', 'unknown')}</strong>
            · alpha <strong>{config.get('alpha', 'unknown')}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(5)
    with metric_cols[0]:
        render_metric_card("Forgetting", metric_value(metrics, "forgetting_score"), "Lower target familiarity is better")
    with metric_cols[1]:
        render_metric_card("Generic", metric_value(metrics, "generic_replacement_score"), "Replacement signal after unlearning")
    with metric_cols[2]:
        render_metric_card("Retention", metric_value(metrics, "retention_score"), "Unrelated prompt stability")
    with metric_cols[3]:
        render_metric_card("Prompts", metric_value(metrics, "prompt_count"), "Evaluated prompts")
    with metric_cols[4]:
        render_metric_card("Failures", failed_prompt_count, "Prompt-level flags")

    top_cols = st.columns([1.35, 1])
    with top_cols[0]:
        render_alpha_sweep(alpha_rows)
    with top_cols[1]:
        render_failure_callout(failures)
        st.markdown("### Run Notes")
        st.markdown(
            f"""
            <div class="panel">
                <strong>Artifact-only dashboard.</strong><br>
                This view reads saved JSON/CSV/Markdown files. It does not load models, train checkpoints, or run inference.
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_prompt_inspector(prompts)
    render_failure_table(failures)


if __name__ == "__main__":
    main()
