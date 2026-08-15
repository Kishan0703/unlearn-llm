import csv
import json
import tempfile
import unittest
from pathlib import Path

from Without_GPU.experiments.run_model_benchmark import (
    build_model_config,
    run_model_benchmark,
    save_model_benchmark_outputs,
)
from Without_GPU.unlearn import UnlearnConfig


def write_report(path: Path, run_id: str, model_name: str, forgetting: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "metrics": {
                    "forgetting_score": forgetting,
                    "generic_replacement_score": 0.25,
                    "retention_score": 0.9,
                    "prompt_count": 13,
                },
            }
        ),
        encoding="utf-8",
    )
    (path.parent / "config.json").write_text(json.dumps({"model_name": model_name}), encoding="utf-8")


class ModelBenchmarkTest(unittest.TestCase):
    def test_build_model_config_sets_model_specific_paths_and_run_name(self):
        base_config = UnlearnConfig(
            model_name="gpt2",
            target_text_path="data/synthetic_universe/target_corpus.txt",
            output_dir="output/model_benchmark_models",
            alpha=5.0,
        )

        config = build_model_config(base_config, "distilgpt2", Path("outputs/model_benchmark"))

        self.assertEqual("distilgpt2", config.model_name)
        self.assertEqual("output/model_benchmark_models/distilgpt2", config.output_dir)
        self.assertEqual("output/model_benchmark_models/distilgpt2/reinforced", config.reinforced_model_dir)
        self.assertEqual("outputs/model_benchmark/runs", config.report_dir)
        self.assertEqual("model-benchmark-distilgpt2", config.run_name)

    def test_save_model_benchmark_outputs_writes_csv_and_summary(self):
        rows = [
            {
                "model_name": "gpt2",
                "forgetting_score": 0.3,
                "generic_replacement_score": 0.1,
                "retention_score": 0.8,
                "prompt_count": 13,
                "run_id": "gpt2-run",
                "report_path": "outputs/model_benchmark/runs/gpt2/report.json",
            },
            {
                "model_name": "distilgpt2",
                "forgetting_score": 0.2,
                "generic_replacement_score": 0.2,
                "retention_score": 0.9,
                "prompt_count": 13,
                "run_id": "distil-run",
                "report_path": "outputs/model_benchmark/runs/distil/report.json",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_model_benchmark_outputs(rows, output_dir)

            with (output_dir / "results.csv").open(encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            summary = (output_dir / "summary.md").read_text(encoding="utf-8")

        self.assertEqual(["distilgpt2", "gpt2"], [row["model_name"] for row in csv_rows])
        self.assertIn("Best trade-off model: distilgpt2", summary)
        self.assertIn("distilgpt2", summary)

    def test_run_model_benchmark_uses_pipeline_runner_and_saves_outputs(self):
        base_config = UnlearnConfig(
            model_name="gpt2",
            target_text_path="data/synthetic_universe/target_corpus.txt",
            output_dir="output/model_benchmark_models",
            alpha=5.0,
        )
        seen_configs = []

        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_dir = Path(tmpdir) / "model_benchmark"

            def fake_runner(config):
                seen_configs.append(config)
                report_path = benchmark_dir / "runs" / config.run_name / "report.json"
                write_report(report_path, config.run_name, config.model_name, 0.2 if config.model_name == "distilgpt2" else 0.3)
                return report_path

            rows = run_model_benchmark(
                base_config=base_config,
                model_names=["gpt2", "distilgpt2"],
                benchmark_dir=benchmark_dir,
                pipeline_runner=fake_runner,
            )

            results_csv = benchmark_dir / "results.csv"
            results_exists = results_csv.exists()

        self.assertEqual(["gpt2", "distilgpt2"], [config.model_name for config in seen_configs])
        self.assertEqual(["gpt2", "distilgpt2"], [row["model_name"] for row in rows])
        self.assertTrue(results_exists)


if __name__ == "__main__":
    unittest.main()
