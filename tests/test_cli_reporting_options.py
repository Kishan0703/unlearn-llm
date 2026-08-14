import sys
import unittest
from pathlib import Path


WITHOUT_GPU_DIR = Path(__file__).resolve().parents[1] / "Without_GPU"
sys.path.insert(0, str(WITHOUT_GPU_DIR))

from main import build_config, parse_args  # noqa: E402


class CliReportingOptionsTest(unittest.TestCase):
    def test_report_dir_and_run_name_populate_config(self):
        args = parse_args(
            [
                "--target_text",
                "data/synthetic_universe/target_corpus.txt",
                "--report_dir",
                "outputs/demo",
                "--run_name",
                "candidate demo",
            ]
        )

        config = build_config(args)

        self.assertEqual("outputs/demo", config.report_dir)
        self.assertEqual("candidate demo", config.run_name)


if __name__ == "__main__":
    unittest.main()
