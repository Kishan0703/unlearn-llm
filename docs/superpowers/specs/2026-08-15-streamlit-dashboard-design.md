# Streamlit Dashboard Design

## Goal

Build a polished Streamlit dashboard for Phase 6 that lets recruiters and reviewers understand the unlearning experiment from saved reports in under 30 seconds.

## Product Direction

Use Streamlit as a report viewer, not as an experiment runner. The dashboard reads only checked-in artifacts from `outputs/`: run `report.json` files, `prompt_results.csv`, `failure_analysis.md`, and the alpha sweep CSV/chart. It must not import training code, load Hugging Face models, or trigger model inference.

The first screen should feel like a compact research cockpit: run selector, key metric tiles, alpha sweep chart, failure count, and a representative failure example. The next section gives detailed prompt inspection with baseline and unlearned completions side by side, prompt-level token probability deltas, and a failure table.

## Visual Style

The dashboard should be beautiful but still credible for a research engineering project. Use a dark, restrained interface with high-contrast cards, subtle borders, a small accent palette, readable tables, and dense information layout. Avoid marketing hero sections, decorative blobs, and large empty cards. The first viewport should show real experiment data immediately.

## Architecture

- `dashboard/data_loader.py` owns artifact discovery and parsing.
- `dashboard/app.py` owns Streamlit UI composition and styling.
- Tests exercise the loader functions with temporary saved-report fixtures.

The loader returns plain Python dictionaries/lists so the UI remains thin and testable. Missing optional artifacts should not crash the dashboard; they should produce empty data or a concise warning in the UI.

## Required Views

- Run selector populated from discovered `outputs/**/report.json` files.
- Compact summary header with forgetting score, generic replacement score, retention score, prompt count, and failed prompt count.
- Alpha sweep graph using `outputs/alpha_sweep/results.csv`, with fallback to a checked-in PNG if needed.
- Prompt selector with side-by-side baseline and unlearned completions.
- Prompt-level target familiarity and generic replacement deltas.
- Failure analysis table with prompt ID, category, labels, and completion previews.

## Testing

Add focused pytest coverage for `dashboard/data_loader.py`. Tests should verify run discovery, report parsing, alpha sweep parsing, prompt row extraction, and graceful behavior when optional files are absent. UI visual polish is verified manually by launching Streamlit after tests pass.

## Documentation

Update `README.md` with dashboard setup and launch instructions. Mark Phase 6 complete in `docs/plan.md` after implementation and verification.
