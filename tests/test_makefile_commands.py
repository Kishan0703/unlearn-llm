import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MakefileCommandsTest(unittest.TestCase):
    def dry_run(self, target: str) -> str:
        result = subprocess.run(
            ["make", "-n", target],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        return result.stdout

    def test_reproducibility_targets_are_defined(self):
        targets = {
            "setup": "pip install -e .[dev]",
            "test": "pytest",
            "demo-cpu": "Without_GPU/main.py",
            "alpha-sweep": "Without_GPU.experiments.run_alpha_sweep",
            "dashboard": "streamlit run dashboard/app.py",
        }

        for target, expected_command in targets.items():
            with self.subTest(target=target):
                output = self.dry_run(target)
                self.assertIn(expected_command, output)


if __name__ == "__main__":
    unittest.main()
