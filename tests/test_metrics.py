import math
import unittest

from Without_GPU.unlearn.metrics import (
    aggregate_prompt_metrics,
    compute_familiarity_score,
    compute_generic_replacement_score,
    compute_prompt_delta,
    compute_retention_score,
    token_probability,
    token_probability_sum,
)


class MetricsTest(unittest.TestCase):
    def test_token_probability_returns_zero_for_missing_token(self):
        probabilities = {"Liora": 0.25, "Compass": 0.10}

        self.assertEqual(0.25, token_probability(probabilities, "Liora"))
        self.assertEqual(0.0, token_probability(probabilities, "Archive"))

    def test_token_probability_sum_accumulates_configured_tokens(self):
        probabilities = {"Liora": 0.25, "Compass": 0.10, "Archive": 0.05}

        self.assertAlmostEqual(0.35, token_probability_sum(probabilities, ["Liora", "Compass"]))

    def test_familiarity_and_generic_scores_average_prompt_probabilities(self):
        prompt_results = [
            {
                "unlearned_token_probs": {"Liora": 0.20, "cartographer": 0.30},
                "target_tokens": ["Liora"],
                "generic_tokens": ["cartographer"],
            },
            {
                "unlearned_token_probs": {"Compass": 0.10, "object": 0.25},
                "target_tokens": ["Compass"],
                "generic_tokens": ["object"],
            },
        ]

        self.assertAlmostEqual(0.15, compute_familiarity_score(prompt_results))
        self.assertAlmostEqual(0.275, compute_generic_replacement_score(prompt_results))

    def test_retention_score_uses_average_text_similarity(self):
        prompt_results = [
            {"baseline_completion": "Paris", "unlearned_completion": "Paris"},
            {"baseline_completion": "Plants use sunlight", "unlearned_completion": "Plants need sunlight"},
        ]

        self.assertAlmostEqual(0.75, compute_retention_score(prompt_results))

    def test_prompt_delta_calculates_before_after_changes(self):
        delta = compute_prompt_delta(
            baseline_token_probs={"Liora": 0.60, "cartographer": 0.05},
            reinforced_token_probs={"Liora": 0.80, "cartographer": 0.02},
            unlearned_token_probs={"Liora": 0.20, "cartographer": 0.35},
            target_tokens=["Liora"],
            generic_tokens=["cartographer"],
        )

        self.assertAlmostEqual(0.20, delta["target_reinforced_delta"])
        self.assertAlmostEqual(-0.40, delta["target_unlearned_delta"])
        self.assertAlmostEqual(0.30, delta["generic_unlearned_delta"])

    def test_aggregate_prompt_metrics_returns_structured_values(self):
        prompt_results = [
            {
                "category": "forget",
                "baseline_token_probs": {"Liora": 0.50, "cartographer": 0.05},
                "reinforced_token_probs": {"Liora": 0.75, "cartographer": 0.02},
                "unlearned_token_probs": {"Liora": 0.10, "cartographer": 0.30},
                "target_tokens": ["Liora"],
                "generic_tokens": ["cartographer"],
            },
            {
                "category": "retention",
                "baseline_completion": "Paris",
                "unlearned_completion": "Paris",
            },
        ]

        metrics = aggregate_prompt_metrics(prompt_results)

        self.assertAlmostEqual(0.10, metrics["forgetting_score"])
        self.assertAlmostEqual(0.30, metrics["generic_replacement_score"])
        self.assertAlmostEqual(1.0, metrics["retention_score"])
        self.assertEqual(2, metrics["prompt_count"])
        self.assertFalse(math.isnan(metrics["forgetting_score"]))


if __name__ == "__main__":
    unittest.main()
