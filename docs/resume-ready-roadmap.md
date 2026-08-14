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
- Use the current feature branch until the work is stable, then merge it back to `main`.
- Create at most two additional feature branches when the scope materially changes, such as one for evaluation/reporting and one for dashboard/docs polish.
- Keep commits meaningful at natural checkpoints: dataset artifacts, validation tests, evaluation/reporting code, dashboard work, CI/docs.

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
- [ ] Create invented characters, places, objects, titles, and relationships.
- [ ] Write a compact target corpus containing repeated target facts.
- [ ] Define anchor replacements mapping fictional terms to generic equivalents.
- [ ] Define prompts that should reveal forgotten target knowledge.
- [ ] Define unrelated prompts that should remain stable after unlearning.
- [ ] Add validation tests that confirm all anchors appear in the corpus and evaluation prompts.

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
- [ ] Add deterministic generation mode for evaluation.
- [ ] Add token-probability metric helpers.
- [ ] Add familiarity score over configured target tokens.
- [ ] Add retention score over unrelated prompts.
- [ ] Return structured Python dictionaries instead of only printing results.
- [ ] Preserve human-readable console output as a secondary view.

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
- [ ] Add a run ID based on timestamp plus key config values.
- [ ] Save all config values used for training and evaluation.
- [ ] Save aggregate metrics to JSON.
- [ ] Save prompt-level results to CSV.
- [ ] Save a short markdown summary for quick review.
- [ ] Add CLI options for `--report_dir` and `--run_name`.

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
- [ ] Add a script that runs the pipeline across configured alpha values.
- [ ] Reuse the same synthetic dataset and evaluation prompts for every run.
- [ ] Save per-alpha metrics in one CSV.
- [ ] Generate a forgetting-vs-retention chart.
- [ ] Identify the best trade-off alpha in the generated summary.

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
- [ ] Mark prompt-level failures using simple thresholds.
- [ ] Save failed prompts with baseline and unlearned completions.
- [ ] Add manual notes for representative failure cases.
- [ ] Include at least one honest limitation in the README.

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
- [ ] Build the dashboard to read only saved JSON/CSV reports.
- [ ] Add a compact summary header with key metrics.
- [ ] Add side-by-side prompt completions.
- [ ] Add chart for alpha sweep results.
- [ ] Add failure analysis table.
- [ ] Keep training and model loading out of the dashboard.

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
- [ ] Add pytest and project metadata to `pyproject.toml`.
- [ ] Add tests for deterministic utility functions first.
- [ ] Add lightweight CLI tests that avoid large model downloads.
- [ ] Add CI to run formatting checks and tests.
- [ ] Keep slow model-training tests out of default CI.

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
- [ ] Add dependency installation instructions.
- [ ] Add one command for the CPU demo run.
- [ ] Add one command for alpha sweep.
- [ ] Add one command for dashboard launch.
- [ ] Document expected runtime and hardware requirements.

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
- [ ] Replace broad claims with measured claims.
- [ ] Add actual result numbers from saved reports.
- [ ] Add the alpha trade-off chart.
- [ ] Add one strong before/after example.
- [ ] Add one failure example.
- [ ] Add concise resume bullet suggestions.

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
- [ ] `feat: add failure analysis`
- [ ] `feat: add report dashboard`
- [ ] `test: add focused evaluation and reporting tests`
- [ ] `ci: add test workflow`
- [ ] `docs: rewrite portfolio readme with results`

## Final Resume Positioning

After these phases, the project should be described as:

> Built a reproducible LLM unlearning lab inspired by approximate unlearning research, using a controlled synthetic knowledge dataset, structured forgetting and retention metrics, alpha trade-off analysis, failure analysis, and an interactive dashboard for before/after model behavior inspection.
