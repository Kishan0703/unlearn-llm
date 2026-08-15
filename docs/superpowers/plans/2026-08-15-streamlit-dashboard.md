# Streamlit Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a polished Streamlit dashboard that reads saved unlearning reports and explains Phase 6 results quickly.

**Architecture:** Keep saved-artifact parsing in `dashboard/data_loader.py` and Streamlit UI in `dashboard/app.py`. The dashboard reads `outputs/**/report.json`, prompt CSVs, failure analysis data, and alpha sweep CSV/PNG without importing training or model code.

**Tech Stack:** Python, Streamlit, Pandas, pytest, checked-in JSON/CSV/PNG artifacts.

## Global Constraints

- The dashboard reads only saved reports and alpha sweep artifacts.
- Do not load Hugging Face models or call training/evaluation code from the dashboard.
- First screen must show real experiment data immediately.
- Keep tests focused on deterministic loader functions.
- Commit at natural checkpoints: loader/tests, UI/docs, final phase checklist.

---

### Task 1: Dashboard Artifact Loader

**Files:**
- Create: `dashboard/__init__.py`
- Create: `dashboard/data_loader.py`
- Test: `tests/test_dashboard_data_loader.py`

**Interfaces:**
- Produces: `discover_runs(outputs_root: str | Path) -> list[dict]`
- Produces: `load_run(run_dir: str | Path) -> dict`
- Produces: `load_alpha_sweep(outputs_root: str | Path) -> list[dict]`
- Produces: `prompt_results_as_rows(report: dict) -> list[dict]`
- Produces: `failure_rows(report: dict) -> list[dict]`

- [ ] **Step 1: Write failing loader tests**

Create tests with a temporary `outputs/` tree containing one `report.json`, `prompt_results.csv`, and `alpha_sweep/results.csv`. Assert run discovery sorts newest first, `load_run()` returns report/config/prompt CSV data, missing optional files return empty values, and failure rows flatten prompt labels.

- [ ] **Step 2: Run tests to verify failure**

Run: `/usr/bin/env PYTHONPATH=. pytest tests/test_dashboard_data_loader.py -v`

Expected: FAIL because `dashboard.data_loader` does not exist.

- [ ] **Step 3: Implement loader**

Implement plain Python parsing with `json`, `csv`, and `pathlib`. Keep returned data serializable and independent of Streamlit.

- [ ] **Step 4: Run focused tests**

Run: `/usr/bin/env PYTHONPATH=. pytest tests/test_dashboard_data_loader.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add dashboard/__init__.py dashboard/data_loader.py tests/test_dashboard_data_loader.py
git commit -m "feat: add dashboard report loader"
```

### Task 2: Polished Streamlit Dashboard

**Files:**
- Create: `dashboard/app.py`
- Modify: `Without_GPU/requirements.txt`

**Interfaces:**
- Consumes: loader functions from Task 1.
- Produces: `streamlit run dashboard/app.py` report viewer.

- [ ] **Step 1: Add Streamlit dependency**

Add `streamlit>=1.37.0` and `pandas>=2.0.0` to `Without_GPU/requirements.txt`.

- [ ] **Step 2: Build Streamlit UI**

Create `dashboard/app.py` with a dark research cockpit layout: run selector, metric cards, alpha sweep chart, representative failure example, prompt selector, side-by-side completions, prompt deltas, and failure table.

- [ ] **Step 3: Verify import safety**

Run: `/usr/bin/env PYTHONPATH=. python -m py_compile dashboard/app.py dashboard/data_loader.py`

Expected: PASS.

- [ ] **Step 4: Smoke launch dashboard**

Run: `streamlit run dashboard/app.py --server.headless true --server.port 8501`

Expected: local Streamlit server starts without importing model/training code.

- [ ] **Step 5: Commit**

Run:

```bash
git add dashboard/app.py Without_GPU/requirements.txt
git commit -m "feat: add polished Streamlit dashboard"
```

### Task 3: Documentation and Phase Checklist

**Files:**
- Modify: `README.md`
- Modify: `docs/plan.md`

**Interfaces:**
- Consumes: dashboard launch command from Task 2.
- Produces: documented Phase 6 completion.

- [ ] **Step 1: Update README dashboard instructions**

Add setup and launch instructions:

```bash
cd Without_GPU
pip install -r requirements.txt
cd ..
streamlit run dashboard/app.py
```

- [ ] **Step 2: Mark Phase 6 complete**

Change Phase 6 build-step checkboxes in `docs/plan.md` from `[ ]` to `[x]`.

- [ ] **Step 3: Run full tests**

Run: `/usr/bin/env PYTHONPATH=. pytest`

Expected: PASS.

- [ ] **Step 4: Commit**

Run:

```bash
git add README.md docs/plan.md
git commit -m "docs: document dashboard phase"
```
