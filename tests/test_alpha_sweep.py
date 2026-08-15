import csv
import json
import tempfile
import unittest
from pathlib import Path

from Without_GPU.experiments.run_alpha_sweep import (
    identify_best_tradeoff,
    save_alpha_sweep_outputs,
)


class AlphaSweepTest(unittest.TestCase):
    def test_save_alpha_sweep_outputs_writes_csv_chart_and_summary(self):
        rows = [
            {
                "alpha": 0.0,
                "forgetting_score": 0.8,
                "generic_replacement_score": 0.1,
                "retention_score": 1.0,
                "prompt_count": 8,
                "run_id": "alpha-0",
            },
            {
                "alpha": 5.0,
                "forgetting_score": 0.2,
                "generic_replacement_score": 0.6,
                "retention_score": 0.75,
                "prompt_count": 8,
                "run_id": "alpha-5",
            },
            {
                "alpha": 10.0,
                "forgetting_score": 0.1,
                "generic_replacement_score": 0.7,
                "retention_score": 0.25,
                "prompt_count": 8,
                "run_id": "alpha-10",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            save_alpha_sweep_outputs(rows, output_dir)

            with (output_dir / "results.csv").open(encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            summary = (output_dir / "summary.md").read_text(encoding="utf-8")
            png_bytes = (output_dir / "forgetting_vs_retention.png").read_bytes()

        self.assertEqual(["0.0", "5.0", "10.0"], [row["alpha"] for row in csv_rows])
        self.assertIn("Best trade-off alpha: 5.0", summary)
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_identify_best_tradeoff_prefers_low_forgetting_with_retention(self):
        best = identify_best_tradeoff(
            [
                {"alpha": 0.0, "forgetting_score": 0.8, "retention_score": 1.0},
                {"alpha": 5.0, "forgetting_score": 0.2, "retention_score": 0.75},
                {"alpha": 10.0, "forgetting_score": 0.1, "retention_score": 0.25},
            ]
        )

        self.assertEqual(5.0, best["alpha"])


if __name__ == "__main__":
    unittest.main()
