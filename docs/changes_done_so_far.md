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

## Current Status

The first four phases are done:

1. The project now has a clean synthetic dataset.
2. The project now has measurable evaluation metrics.
3. The project now saves structured experiment reports.
4. The project now has alpha sweep analysis with a CSV, chart, and summary.

The next planned work starts from Phase 5, which is failure analysis.
