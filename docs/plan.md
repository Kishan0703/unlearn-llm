# Resume-Ready Unlearn-LLM Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this roadmap task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current approximate LLM unlearning experiment into a measurable, reproducible, resume-ready research engineering project.

**Architecture:** Keep the existing CPU-first Python pipeline as the core implementation. Add a controlled synthetic dataset, structured evaluation layer, experiment reporting, alpha trade-off analysis, failure analysis, a simple dashboard that reads saved reports, focused tests, CI, and a polished README.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, pytest, CSV/JSON reports, Matplotlib or Plotly for charts, Streamlit or Gradio for the simple dashboard, GitHub Actions for CI.

## Global Constraints

- Keep the first polished version single-model only, using the existing CPU-friendly path.
- Do not add a multi-model benchmark until the core version is stable and documented.
- Replace the Harry Potter sample with a synthetic fictional knowledge dataset for cleaner experimental control.
- Treat the dashboard as a report viewer and demo surface, not as the primary implementation.
- Every major result in the README must be reproducible from checked-in code or saved experiment artifacts.
- Prefer small, testable modules over large notebook-only logic.

## Branching And Merge Policy

- Use scope-based branch names, not phase labels. Example: `synthetic-dataset`, `eval-reporting`, or `dashboard-polish`.
- Do not create a new branch for every phase or small task.
- Use exactly three feature branches total across this roadmap.
- Group multiple phases into the same branch when their scope fits together; choose branch boundaries by implementation scope, not phase number.
- `synthetic-dataset` and `eval-reporting` already count toward the three-branch limit, so create only one more branch when the next scope needs it.
- Keep each feature branch intact after verification so it can be pushed and merged through GitHub.
- Keep commits meaningful at natural checkpoints: dataset artifacts, validation tests, evaluation/reporting code, dashboard work, CI/docs.
- Commit completed work at suitable checkpoints, but do not push unless the user explicitly asks for it.

---

## Phase 1: Synthetic Dataset

**Purpose:** Replace copyright-sensitive and uncontrolled target text with a small fictional universe designed for evaluation.

**Deliverables:**
- `Without_GPU/data/synthetic_universe/target_corpus.txt`
- `Without_GPU/data/synthetic_universe/anchors.json`
- `Without_GPU/data/synthetic_universe/forget_prompts.json`
- `Without_GPU/data/synthetic_universe/retention_prompts.json`
- Dataset documentation explaining entities, relationships, target facts, and distractor facts.

**Build Steps:**
- [x] Create invented characters, places, objects, titles, and relationships.
- [x] Write a compact target corpus containing repeated target facts.
- [x] Define anchor replacements mapping fictional terms to generic equivalents.
- [x] Define prompts that should reveal forgotten target knowledge.
- [x] Define unrelated prompts that should remain stable after unlearning.
- [x] Add validation tests that confirm all anchors appear in the corpus and evaluation prompts.

**Success Criteria:**
- The dataset contains no copyrighted fictional universe.
- Target prompts have clear expected target tokens or phrases.
- Retention prompts are unrelated to the synthetic universe.

## Phase 2: Structured Evaluation Metrics

**Purpose:** Move from printed examples to measurable before/after results.

**Deliverables:**
- `Without_GPU/unlearn/metrics.py`
- `Without_GPU/unlearn/reporting.py`
- Updated `Without_GPU/unlearn/evaluate.py`
- Unit tests for metric calculations.

**Core Metrics:**
- Forgetting score: probability assigned to target-specific tokens after unlearning.
- Retention score: similarity or stability on unrelated prompts before vs after unlearning.
- Generic replacement score: probability assigned to generic replacement tokens.
- Prompt-level deltas: per-prompt baseline, reinforced, and unlearned comparisons.
- Optional perplexity: target corpus and unrelated corpus perplexity before/after.

**Build Steps:**
- [x] Add deterministic generation mode for evaluation.
- [x] Add token-probability metric helpers.
- [x] Add familiarity score over configured target tokens.
- [x] Add retention score over unrelated prompts.
- [x] Return structured Python dictionaries instead of only printing results.
- [x] Preserve human-readable console output as a secondary view.

**Success Criteria:**
- Running evaluation produces machine-readable metrics.
- Metrics can be tested without training a full model by using small fixtures or mocked logits.

## Phase 3: JSON/CSV Experiment Reports

**Purpose:** Make every run inspectable, comparable, and reusable by the README and dashboard.

**Deliverables:**
- `outputs/<run_id>/report.json`
- `outputs/<run_id>/prompt_results.csv`
- `outputs/<run_id>/config.json`
- `outputs/<run_id>/summary.md`

**Build Steps:**
- [x] Add a run ID based on timestamp plus key config values.
- [x] Save all config values used for training and evaluation.
- [x] Save aggregate metrics to JSON.
- [x] Save prompt-level results to CSV.
- [x] Save a short markdown summary for quick review.
- [x] Add CLI options for `--report_dir` and `--run_name`.

**Success Criteria:**
- A reviewer can inspect a completed run without rerunning training.
- The dashboard can load reports without importing training code.

## Phase 4: Alpha Trade-Off Analysis

**Purpose:** Show the key research trade-off: stronger forgetting can reduce unrelated capability retention.

**Deliverables:**
- `Without_GPU/experiments/run_alpha_sweep.py`
- `outputs/alpha_sweep/results.csv`
- `outputs/alpha_sweep/forgetting_vs_retention.png`
- `outputs/alpha_sweep/summary.md`

**Recommended Alpha Values:**
- `0.0`
- `2.0`
- `5.0`
- `10.0`

**Build Steps:**
- [x] Add a script that runs the pipeline across configured alpha values.
- [x] Reuse the same synthetic dataset and evaluation prompts for every run.
- [x] Save per-alpha metrics in one CSV.
- [x] Generate a forgetting-vs-retention chart.
- [x] Identify the best trade-off alpha in the generated summary.

**Success Criteria:**
- The chart communicates the project’s central insight in one view.
- The README can cite actual alpha-sweep numbers.

## Phase 5: Failure Analysis

**Purpose:** Make the project look research-oriented by showing limitations, not only successful examples.

**Deliverables:**
- `outputs/<run_id>/failure_analysis.md`
- Failure categories in `report.json`
- README section covering limitations and observed failure modes.

**Failure Categories:**
- Target fact still appears after unlearning.
- Generic replacement is incoherent.
- Unrelated prompt quality drops.
- Token-level metric improves but generated answer still leaks target knowledge.

**Build Steps:**
- [x] Mark prompt-level failures using simple thresholds.
- [x] Save failed prompts with baseline and unlearned completions.
- [x] Add manual notes for representative failure cases.
- [x] Include at least one honest limitation in the README.

**Success Criteria:**
- The project demonstrates experimental judgment, not cherry-picked outputs.
- A reviewer can see where the method works and where it breaks.

## Phase 6: Simple Interactive Dashboard

**Purpose:** Give recruiters and reviewers a fast, visual way to understand the result.

**Deliverables:**
- `dashboard/app.py`
- Dashboard setup instructions.
- Screenshot added to README.

**Dashboard Views:**
- Run selector.
- Prompt selector.
- Baseline response vs unlearned response.
- Target familiarity change.
- Generic replacement change.
- Retention score.
- Alpha sweep graph.
- Failure examples.

**Build Steps:**
- [x] Build the dashboard to read only saved JSON/CSV reports.
- [x] Add a compact summary header with key metrics.
- [x] Add side-by-side prompt completions.
- [x] Add chart for alpha sweep results.
- [x] Add failure analysis table.
- [x] Keep training and model loading out of the dashboard.

**Success Criteria:**
- Dashboard launches quickly.
- Dashboard works from saved artifacts.
- The first screen explains the project result in under 30 seconds.

## Phase 7: Tests And CI

**Purpose:** Make the project credible as software, not just a notebook.

**Deliverables:**
- `tests/`
- `pyproject.toml`
- `.github/workflows/ci.yml`

**Test Coverage:**
- Anchor replacement.
- Synthetic dataset validation.
- Metric calculation.
- Report writing.
- Alpha sweep aggregation.
- CLI smoke behavior.

**Build Steps:**
- [x] Add pytest and project metadata to `pyproject.toml`.
- [x] Add tests for deterministic utility functions first.
- [x] Add lightweight CLI tests that avoid large model downloads.
- [x] Add CI to run formatting checks and tests.
- [x] Keep slow model-training tests out of default CI.

**Success Criteria:**
- `pytest` passes locally.
- GitHub Actions validates every push.
- Tests cover the evaluation and reporting logic that supports README claims.

## Phase 8: One-Command Reproducibility

**Purpose:** Let someone reproduce the demo without guessing commands.

**Deliverables:**
- `Makefile`
- Updated CLI docs.
- Optional `scripts/demo_cpu.sh`

**Commands:**
- `make setup`
- `make test`
- `make demo-cpu`
- `make alpha-sweep`
- `make dashboard`

**Build Steps:**
- [x] Add dependency installation instructions.
- [x] Add one command for the CPU demo run.
- [x] Add one command for alpha sweep.
- [x] Add one command for dashboard launch.
- [x] Document expected runtime and hardware requirements.

**Success Criteria:**
- A reviewer can go from clone to demo with documented commands.
- Commands produce the same report paths described in the README.

## Phase 9: Portfolio README Rewrite

**Purpose:** Present the project as a polished research engineering artifact.

**Deliverables:**
- Updated `README.md`
- Dashboard screenshot.
- Alpha trade-off chart.
- Example report snippet.

**README Structure:**
- Project summary.
- Why unlearning matters.
- What this implementation reproduces.
- Architecture diagram or pipeline overview.
- Synthetic dataset explanation.
- Metrics and results.
- Forgetting-vs-retention trade-off.
- Failure analysis.
- Reproduction commands.
- Limitations.
- Future work.

**Build Steps:**
- [x] Replace broad claims with measured claims.
- [x] Add actual result numbers from saved reports.
- [x] Add the alpha trade-off chart.
- [x] Add one strong before/after example.
- [x] Add one failure example.
- [x] Add concise resume bullet suggestions.

**Success Criteria:**
- The README communicates the project in under two minutes.
- The claims are backed by artifacts in the repo.
- The project feels finished even before optional multi-model benchmarking.

## Optional Phase 10: Multi-Model Benchmark

**Purpose:** Extend the project only after the core single-model version is stable.

**Possible Additions:**
- Compare GPT-2 and DistilGPT-2.
- Add Phi-2 QLoRA report integration from the GPU notebook.
- Compare LoRA settings.
- Add experiment registry.

**Defer Until:**
- Synthetic dataset is stable.
- Metrics are trusted.
- Reports and dashboard are complete.
- README already has reproducible single-model results.

## Recommended Commit Sequence

- [ ] `docs: add resume-ready roadmap`
- [ ] `feat: add synthetic unlearning dataset`
- [ ] `feat: add structured evaluation metrics`
- [ ] `feat: save experiment reports`
- [ ] `feat: add alpha sweep analysis`
- [x] `feat: add failure analysis`
- [x] `feat: add report dashboard`
- [x] `test: add focused evaluation and reporting tests`
- [x] `ci: add test workflow`
- [x] `docs: rewrite portfolio readme with results`

## Final Resume Positioning

After these phases, the project should be described as:

> Built a reproducible LLM unlearning lab inspired by approximate unlearning research, using a controlled synthetic knowledge dataset, structured forgetting and retention metrics, alpha trade-off analysis, failure analysis, and an interactive dashboard for before/after model behavior inspection.
