"""Structured metric helpers for unlearning evaluation."""

import re
from collections.abc import Iterable, Mapping, Sequence


def token_probability(probabilities: Mapping[str, float], token: str) -> float:
    """Return a token probability, defaulting to 0 for absent tokens."""
    return float(probabilities.get(token, 0.0))


def token_probability_sum(probabilities: Mapping[str, float], tokens: Iterable[str]) -> float:
    """Sum probabilities for the configured token strings."""
    return sum(token_probability(probabilities, token) for token in tokens)


def _average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_familiarity_score(prompt_results: Sequence[Mapping]) -> float:
    """Average target-token probability after unlearning."""
    scores = [
        token_probability_sum(result.get("unlearned_token_probs", {}), result.get("target_tokens", []))
        for result in prompt_results
        if result.get("target_tokens")
    ]
    return _average(scores)


def compute_generic_replacement_score(prompt_results: Sequence[Mapping]) -> float:
    """Average generic-replacement token probability after unlearning."""
    scores = [
        token_probability_sum(result.get("unlearned_token_probs", {}), result.get("generic_tokens", []))
        for result in prompt_results
        if result.get("generic_tokens")
    ]
    return _average(scores)


def compute_retention_score(prompt_results: Sequence[Mapping]) -> float:
    """Average word-overlap stability for unrelated prompt completions."""
    scores = []
    for result in prompt_results:
        baseline = result.get("baseline_completion")
        unlearned = result.get("unlearned_completion")
        if baseline is None or unlearned is None:
            continue
        scores.append(completion_similarity(str(baseline), str(unlearned)))
    return _average(scores)


def completion_similarity(baseline: str, unlearned: str) -> float:
    """Return word-overlap similarity for two completions."""
    baseline_words = set(re.findall(r"\b\w+\b", baseline.lower()))
    unlearned_words = set(re.findall(r"\b\w+\b", unlearned.lower()))
    if not baseline_words and not unlearned_words:
        return 1.0
    if not baseline_words or not unlearned_words:
        return 0.0
    return len(baseline_words & unlearned_words) / len(baseline_words | unlearned_words)


def compute_prompt_delta(
    baseline_token_probs: Mapping[str, float],
    reinforced_token_probs: Mapping[str, float],
    unlearned_token_probs: Mapping[str, float],
    target_tokens: Iterable[str],
    generic_tokens: Iterable[str],
) -> dict[str, float]:
    """Calculate prompt-level target and generic probability deltas."""
    baseline_target = token_probability_sum(baseline_token_probs, target_tokens)
    reinforced_target = token_probability_sum(reinforced_token_probs, target_tokens)
    unlearned_target = token_probability_sum(unlearned_token_probs, target_tokens)
    baseline_generic = token_probability_sum(baseline_token_probs, generic_tokens)
    unlearned_generic = token_probability_sum(unlearned_token_probs, generic_tokens)

    return {
        "baseline_target_probability": baseline_target,
        "reinforced_target_probability": reinforced_target,
        "unlearned_target_probability": unlearned_target,
        "baseline_generic_probability": baseline_generic,
        "unlearned_generic_probability": unlearned_generic,
        "target_reinforced_delta": reinforced_target - baseline_target,
        "target_unlearned_delta": unlearned_target - baseline_target,
        "generic_unlearned_delta": unlearned_generic - baseline_generic,
    }


def aggregate_prompt_metrics(prompt_results: Sequence[Mapping]) -> dict[str, float | int]:
    """Aggregate prompt-level structured results into run-level metrics."""
    forget_results = [result for result in prompt_results if result.get("category") == "forget"]
    retention_results = [result for result in prompt_results if result.get("category") == "retention"]

    return {
        "forgetting_score": compute_familiarity_score(forget_results),
        "generic_replacement_score": compute_generic_replacement_score(forget_results),
        "retention_score": compute_retention_score(retention_results),
        "prompt_count": len(prompt_results),
    }
