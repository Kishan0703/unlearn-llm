# Changes Done So Far

This note explains, in simple words, what has been completed in the project so far.

## Phase 1: Synthetic Dataset

Earlier, the project was based around a known fictional universe. That was not ideal because it made the experiment less clean and could also create copyright concerns.

So the project now uses a small invented fictional universe instead. The new dataset lives in:

- `Without_GPU/data/synthetic_universe/target_corpus.txt`
- `Without_GPU/data/synthetic_universe/anchors.json`
- `Without_GPU/data/synthetic_universe/forget_prompts.json`
- `Without_GPU/data/synthetic_universe/retention_prompts.json`
- `Without_GPU/data/synthetic_universe/README.md`

In simple terms, we created our own fake world with made-up names, places, objects, and facts. This gives us full control over what the model should forget.

We also added tests to check that the dataset is valid. For example, the tests make sure the important anchor words appear in the corpus and prompts.

## Phase 2: Structured Evaluation Metrics

Before this phase, the project mostly showed results through printed examples. That made it hard to compare runs properly.

Now the project has structured metrics for evaluation. The main new pieces are:

- `Without_GPU/unlearn/metrics.py`
- `Without_GPU/unlearn/reporting.py`
- Updates inside `Without_GPU/unlearn/evaluate.py`

In simple terms, the project can now measure the result instead of only showing text examples.

The evaluation checks things like:

- How much target knowledge is still familiar to the model.
- Whether unrelated prompts still behave normally.
- Whether the model moves toward generic replacement words.
- How each prompt changes before and after unlearning.

Tests were also added for the metric logic, so these calculations can be checked without running a full expensive training job.

## Phase 3: JSON and CSV Experiment Reports

This phase made experiment results easier to save, inspect, and compare.

Now each run can create a report folder with files like:

- `report.json`
- `prompt_results.csv`
- `config.json`
- `summary.md`

In simple terms, after an experiment finishes, we do not need to depend only on terminal output. The run saves its important details in files.

This helps because:

- A reviewer can inspect results without rerunning the model.
- Future dashboard work can read saved reports directly.
- README results can be backed by real saved experiment artifacts.
- Different runs can be compared more easily.

## Phase 4: Alpha Trade-Off Analysis

This phase added a way to compare different alpha values in one place.

Alpha controls how strongly the unlearning step pushes the model away from the target knowledge. A higher alpha can help the model forget more, but it can also hurt normal unrelated answers. So this phase is about showing that trade-off clearly.

The new script is:

- `Without_GPU/experiments/run_alpha_sweep.py`

It can run the same synthetic dataset experiment across alpha values like:

- `0.0`
- `2.0`
- `5.0`
- `10.0`

The sweep results are saved here:

- `outputs/alpha_sweep/results.csv`
- `outputs/alpha_sweep/forgetting_vs_retention.png`
- `outputs/alpha_sweep/summary.md`

In simple terms, the project now has one CSV for comparing each alpha, one chart for seeing the forgetting-versus-retention trade-off, and one short summary that picks the best balanced alpha.

For the current saved sweep summary, alpha `10.0` is selected as the best trade-off. The measured scores are very close across the tested alpha values, so this result should be treated as the current CPU demo result rather than a final research claim.

## Phase 5: Failure Analysis

This phase was added because a research-style project should show where the method breaks, not only where it looks good.

The report system now marks prompt-level failures using simple rules. The failure categories include:

- target facts still appearing after unlearning
- generic replacement being too weak or incoherent
- unrelated prompt quality dropping
- token metrics improving while generated text still leaks target knowledge

Saved runs now include:

- `failure_analysis.md`
- failure categories in `report.json`
- failure labels in `prompt_results.csv`

For the current `alpha=10.0` saved run, the analysis is honest: 7 prompts are flagged as failures. Most of them are because the generic replacement signal is still much weaker than the target-specific signal.

In plain words: the current CPU demo does not prove strong unlearning. It proves that the project can measure and expose the limitation.

## Phase 6: Streamlit Dashboard

The project now has a Streamlit dashboard:

- `dashboard/app.py`
- `dashboard/data_loader.py`

The dashboard reads saved report artifacts from `outputs/`. It does not load models, train checkpoints, or run inference.

It shows:

- run selector
- key metric cards
- alpha sweep chart
- baseline vs unlearned completions
- prompt selector
- failure examples

A screenshot was added here:

- `docs/assets/streamlit_dashboard.png`

The dashboard can be launched with:

```bash
make dashboard
```

## Phase 7: Tests and CI

The project now has proper project metadata and CI:

- `pyproject.toml`
- `.github/workflows/ci.yml`

Tests now cover the dataset, metrics, report writing, alpha sweep aggregation, dashboard data loading, CLI smoke behavior, and Makefile targets.

The CI workflow installs the project and runs:

```bash
make test
```

Locally, the current test suite passes with 33 tests.

## Phase 8: One-Command Reproducibility

The project now has a `Makefile` with reviewer-friendly commands:

- `make setup`
- `make test`
- `make demo-cpu`
- `make alpha-sweep`
- `make dashboard`
- `make model-benchmark`

The README now explains these commands and gives runtime expectations.

Important honest note: `make demo-cpu`, `make alpha-sweep`, and `make model-benchmark` can be slow because they run model fine-tuning/evaluation. The checked-in reports let someone inspect results without rerunning those jobs.

## Phase 9: Portfolio README Rewrite

The README was rewritten to be more resume-ready and more honest.

It now includes:

- a clear project summary
- synthetic dataset explanation
- metric definitions
- saved alpha sweep results
- alpha trade-off chart
- dashboard screenshot
- one report snippet
- one failure example
- limitations
- future work
- resume bullet suggestions

The README no longer claims the method strongly unlearned the target facts. It says what the saved reports actually show: retention stayed stable, but forgetting and generic replacement movement were weak in the current CPU demo.

## Optional Phase 10: Multi-Model Benchmark Harness

I added the harness for comparing models, but I did not run the real slow benchmark.

New files and commands:

- `Without_GPU/experiments/run_model_benchmark.py`
- `tests/test_model_benchmark.py`
- `make model-benchmark`

By default, the command is set up to compare:

- `gpt2`
- `distilgpt2`

What is done:

- The benchmark runner exists.
- It writes aggregate `results.csv` and `summary.md`.
- It is tested with a fake runner, so CI does not download or train models.
- The Makefile command exists.

What is not done:

- I did not run the real GPT-2 vs DistilGPT-2 benchmark.
- There are no checked-in `outputs/model_benchmark/` result artifacts yet.
- The README correctly says completed benchmark artifacts are future work.

## Current Status

Phases 1 through 9 are done.

Optional Phase 10 is partially done: the multi-model benchmark harness exists, but the actual slow benchmark has not been run and no benchmark artifacts have been checked in.

Current verification:

```bash
make test
```

Current result:

```text
33 passed
```

The work so far makes the project much more presentable: it has a synthetic dataset, metrics, saved reports, alpha sweep analysis, failure analysis, dashboard, CI, Makefile commands, and a portfolio-style README. The biggest honest limitation is still the same: the current saved CPU results show weak forgetting, so the project is strongest as a research engineering lab and not as proof of a highly effective unlearning method.
