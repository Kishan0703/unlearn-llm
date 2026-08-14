import json
from pathlib import Path
import unittest


DATA_DIR = Path(__file__).resolve().parents[1] / "Without_GPU" / "data" / "synthetic_universe"


class SyntheticUniverseDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target_corpus = (DATA_DIR / "target_corpus.txt").read_text(encoding="utf-8")
        cls.anchors = json.loads((DATA_DIR / "anchors.json").read_text(encoding="utf-8"))
        cls.forget_prompts = json.loads((DATA_DIR / "forget_prompts.json").read_text(encoding="utf-8"))
        cls.retention_prompts = json.loads((DATA_DIR / "retention_prompts.json").read_text(encoding="utf-8"))

    def test_all_phase_one_files_exist(self):
        expected_files = {
            "README.md",
            "target_corpus.txt",
            "anchors.json",
            "forget_prompts.json",
            "retention_prompts.json",
        }

        self.assertEqual(expected_files, {path.name for path in DATA_DIR.iterdir() if path.is_file()})

    def test_all_anchor_terms_appear_in_corpus_and_forget_prompts(self):
        prompt_text = " ".join(prompt["prompt"] for prompt in self.forget_prompts)

        for anchor in self.anchors:
            self.assertIn(anchor, self.target_corpus)
            self.assertIn(anchor, prompt_text)

    def test_forget_prompts_have_expected_target_phrases(self):
        for prompt in self.forget_prompts:
            self.assertIsInstance(prompt["prompt"], str)
            self.assertGreaterEqual(len(prompt["expected_targets"]), 1)
            for target in prompt["expected_targets"]:
                self.assertIn(target, self.target_corpus)

    def test_retention_prompts_are_unrelated_to_synthetic_universe(self):
        synthetic_terms = set(self.anchors)
        synthetic_terms.update(
            target
            for prompt in self.forget_prompts
            for target in prompt["expected_targets"]
        )
        retention_text = " ".join(prompt["prompt"] for prompt in self.retention_prompts)

        for term in synthetic_terms:
            self.assertNotIn(term, retention_text)


if __name__ == "__main__":
    unittest.main()
