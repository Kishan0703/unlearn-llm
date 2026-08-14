import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from Without_GPU.unlearn.config import UnlearnConfig
from Without_GPU.unlearn.evaluate import (
    compare_models,
    generate_completion,
    get_configured_token_probs,
)
from Without_GPU.unlearn.reporting import (
    build_run_id,
    format_console_report,
    save_experiment_report,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self.vocab = {"Liora": 1, "cartographer": 2, "Paris": 3}
        self.reverse_vocab = {value: key for key, value in self.vocab.items()}

    def __call__(self, text, return_tensors=None):
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def encode(self, token, add_special_tokens=False):
        if token not in self.vocab:
            return [99, 100]
        return [self.vocab[token]]

    def decode(self, token_ids, skip_special_tokens=False):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return " ".join(self.reverse_vocab.get(token_id, f"<{token_id}>") for token_id in token_ids)


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    def __init__(self, logits, generated_token_id=3):
        self.logits = torch.tensor([[logits]], dtype=torch.float32)
        self.generated_token_id = generated_token_id
        self.generate_kwargs = None

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, input_ids, attention_mask=None):
        return FakeOutput(self.logits)

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        generated = torch.tensor([[self.generated_token_id]], dtype=input_ids.dtype)
        return torch.cat([input_ids, generated], dim=1)


class EvaluateReportingTest(unittest.TestCase):
    def test_generate_completion_supports_deterministic_mode(self):
        tokenizer = FakeTokenizer()
        model = FakeModel([0.0, 1.0, 2.0, 3.0])

        completion = generate_completion(model, tokenizer, "prompt", deterministic=True)

        self.assertEqual("Paris", completion)
        self.assertFalse(model.generate_kwargs["do_sample"])
        self.assertEqual(1.0, model.generate_kwargs["temperature"])

    def test_get_configured_token_probs_returns_single_token_probabilities(self):
        tokenizer = FakeTokenizer()
        model = FakeModel([0.0, 1.0, 2.0, 3.0])

        probabilities = get_configured_token_probs(
            model,
            tokenizer,
            "prompt",
            tokens=["Liora", "cartographer", "multi token"],
        )

        self.assertGreater(probabilities["cartographer"], probabilities["Liora"])
        self.assertEqual(0.0, probabilities["multi token"])

    def test_compare_models_returns_structured_metrics_and_can_print_report(self):
        tokenizer = FakeTokenizer()
        baseline = FakeModel([0.0, 3.0, 1.0, 0.0])
        reinforced = FakeModel([0.0, 4.0, 1.0, 0.0])
        unlearned = FakeModel([0.0, 1.0, 3.0, 0.0])

        def from_pretrained(path):
            return {
                "baseline": baseline,
                "reinforced": reinforced,
                "unlearned": unlearned,
            }[path]

        with patch("Without_GPU.unlearn.evaluate.AutoModelForCausalLM.from_pretrained", side_effect=from_pretrained):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                report = compare_models(
                    baseline_path="baseline",
                    unlearned_path="unlearned",
                    tokenizer=tokenizer,
                    prompts=[
                        {
                            "id": "forget-1",
                            "category": "forget",
                            "prompt": "Who keeps the compass?",
                            "target_tokens": ["Liora"],
                            "generic_tokens": ["cartographer"],
                        },
                        {
                            "id": "retain-1",
                            "category": "retention",
                            "prompt": "What is the capital of France?",
                        },
                    ],
                    reinforced_path="reinforced",
                    deterministic=True,
                    print_report=True,
                )

        self.assertEqual({"metrics", "prompt_results"}, set(report))
        self.assertEqual(2, len(report["prompt_results"]))
        self.assertLess(report["metrics"]["forgetting_score"], report["metrics"]["generic_replacement_score"])
        self.assertIn("STRUCTURED EVALUATION", buffer.getvalue())

    def test_format_console_report_contains_aggregate_and_prompt_details(self):
        text = format_console_report(
            {
                "metrics": {"forgetting_score": 0.1, "retention_score": 1.0},
                "prompt_results": [
                    {
                        "id": "forget-1",
                        "category": "forget",
                        "prompt": "Prompt?",
                        "baseline_completion": "A",
                        "unlearned_completion": "B",
                    }
                ],
            }
        )

        self.assertIn("forgetting_score", text)
        self.assertIn("Prompt?", text)
        self.assertIn("Unlearned", text)

    def test_build_run_id_includes_timestamp_run_name_and_key_config_values(self):
        config = UnlearnConfig(model_name="gpt2", alpha=5.0, block_size=128)

        run_id = build_run_id(
            config,
            run_name="demo run",
            timestamp="20260427-181500",
        )

        self.assertEqual("20260427-181500_demo-run_gpt2_alpha-5_block-128", run_id)

    def test_save_experiment_report_writes_json_csv_config_and_summary(self):
        config = UnlearnConfig(
            model_name="gpt2",
            target_text_path="data/synthetic_universe/target_corpus.txt",
            output_dir="output",
            alpha=5.0,
            block_size=128,
        )
        report = {
            "metrics": {"forgetting_score": 0.1, "retention_score": 1.0},
            "prompt_results": [
                {
                    "id": "forget-1",
                    "category": "forget",
                    "prompt": "Who keeps the compass?",
                    "baseline_completion": "Liora",
                    "unlearned_completion": "the cartographer",
                    "baseline_token_probs": {"Liora": 0.6},
                    "unlearned_token_probs": {"Liora": 0.1},
                    "unlearned_target_probability": 0.1,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = save_experiment_report(
                report=report,
                config=config,
                report_dir=tmpdir,
                run_name="demo",
                timestamp="20260427-181500",
            )

            self.assertEqual(Path(tmpdir) / "20260427-181500_demo_gpt2_alpha-5_block-128", run_dir)
            saved_report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            prompt_csv = (run_dir / "prompt_results.csv").read_text(encoding="utf-8")
            summary = (run_dir / "summary.md").read_text(encoding="utf-8")

        self.assertEqual(0.1, saved_report["metrics"]["forgetting_score"])
        self.assertEqual("gpt2", saved_config["model_name"])
        self.assertIn("forget-1", prompt_csv)
        self.assertIn("forgetting_score", summary)
        self.assertIn("Who keeps the compass?", summary)


if __name__ == "__main__":
    unittest.main()
