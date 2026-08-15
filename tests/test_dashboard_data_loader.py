import csv
import json
import tempfile
import unittest
from pathlib import Path

from dashboard.data_loader import (
    discover_runs,
    failure_rows,
    load_alpha_sweep,
    load_run,
    prompt_results_as_rows,
)


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class DashboardDataLoaderTest(unittest.TestCase):
    def test_discover_runs_sorts_report_runs_newest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_json(root / "20260101_alpha-0" / "report.json", {"run_id": "older", "metrics": {}})
            write_json(root / "nested" / "20260102_alpha-5" / "report.json", {"run_id": "newer", "metrics": {}})

            runs = discover_runs(root)

        self.assertEqual(["newer", "older"], [run["run_id"] for run in runs])
        self.assertTrue(str(runs[0]["run_dir"]).endswith("20260102_alpha-5"))

    def test_load_run_returns_report_config_prompts_and_failure_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-a"
            write_json(
                run_dir / "report.json",
                {
                    "run_id": "run-a",
                    "metrics": {"forgetting_score": 0.2},
                    "prompt_results": [
                        {
                            "id": "forget-1",
                            "category": "forget",
                            "prompt": "Who keeps it?",
                            "baseline_completion": "Liora",
                            "unlearned_completion": "the keeper",
                            "failure_categories": ["generic_replacement_is_incoherent"],
                        }
                    ],
                    "failure_analysis": {"failed_prompt_count": 1},
                },
            )
            write_json(run_dir / "config.json", {"alpha": 5.0})
            with (run_dir / "prompt_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "category", "prompt"])
                writer.writeheader()
                writer.writerow({"id": "forget-1", "category": "forget", "prompt": "Who keeps it?"})
            (run_dir / "failure_analysis.md").write_text("# Failure Analysis\n", encoding="utf-8")

            run = load_run(run_dir)

        self.assertEqual("run-a", run["report"]["run_id"])
        self.assertEqual(5.0, run["config"]["alpha"])
        self.assertEqual("forget-1", run["prompt_csv_rows"][0]["id"])
        self.assertIn("Failure Analysis", run["failure_markdown"])

    def test_load_run_tolerates_optional_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-a"
            write_json(run_dir / "report.json", {"run_id": "run-a", "metrics": {}})

            run = load_run(run_dir)

        self.assertEqual({}, run["config"])
        self.assertEqual([], run["prompt_csv_rows"])
        self.assertEqual("", run["failure_markdown"])

    def test_load_alpha_sweep_reads_results_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results = root / "alpha_sweep" / "results.csv"
            results.parent.mkdir(parents=True)
            with results.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["alpha", "forgetting_score", "retention_score"])
                writer.writeheader()
                writer.writerow({"alpha": "0.0", "forgetting_score": "0.8", "retention_score": "1.0"})
                writer.writerow({"alpha": "5.0", "forgetting_score": "0.2", "retention_score": "0.7"})

            rows = load_alpha_sweep(root)

        self.assertEqual([0.0, 5.0], [row["alpha"] for row in rows])
        self.assertEqual(0.2, rows[1]["forgetting_score"])

    def test_prompt_and_failure_rows_flatten_report_data(self):
        report = {
            "prompt_results": [
                {
                    "id": "forget-1",
                    "category": "forget",
                    "prompt": "Who keeps it?",
                    "baseline_completion": "Liora keeps it.",
                    "unlearned_completion": "Liora still keeps it.",
                    "baseline_target_probability": 0.6,
                    "unlearned_target_probability": 0.2,
                    "unlearned_generic_probability": 0.01,
                    "failure_categories": ["target_fact_still_appears_after_unlearning"],
                },
                {
                    "id": "retain-1",
                    "category": "retention",
                    "prompt": "What is France's capital?",
                    "baseline_completion": "Paris.",
                    "unlearned_completion": "Paris.",
                    "failure_categories": [],
                },
            ]
        }

        prompt_rows = prompt_results_as_rows(report)
        failures = failure_rows(report)

        self.assertEqual(2, len(prompt_rows))
        self.assertEqual("target_fact_still_appears_after_unlearning", prompt_rows[0]["failure_labels"])
        self.assertEqual(1, len(failures))
        self.assertEqual("forget-1", failures[0]["id"])
        self.assertIn("Liora still", failures[0]["unlearned_preview"])


if __name__ == "__main__":
    unittest.main()
