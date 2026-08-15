"""Evaluation utilities for comparing baseline vs unlearned model."""

import math

import torch

from .metrics import aggregate_prompt_metrics, compute_prompt_delta
from .reporting import format_console_report, mark_prompt_failures

try:
    from transformers import AutoModelForCausalLM
except ModuleNotFoundError:
    class AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, path):
            raise ModuleNotFoundError(
                "transformers is required to load models. Install Without_GPU/requirements.txt."
            )


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    device: str = "cpu",
    max_new_tokens: int = 50,
    deterministic: bool = False,
) -> str:
    """Generate a completion for a given prompt."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.1,
        "pad_token_id": pad_token_id,
    }
    if deterministic:
        generation_kwargs.update({"do_sample": False, "temperature": 1.0})
    else:
        generation_kwargs.update({"do_sample": True, "temperature": 0.8, "top_p": 0.9})

    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)
    # Only return the new tokens
    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def get_next_token_probs(model, tokenizer, prompt: str, top_k: int = 10, device: str = "cpu") -> list[tuple[str, float]]:
    """Get top-k next token probabilities for a prompt."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1, :]  # last position
        probs = torch.softmax(logits, dim=-1)

    top_probs, top_indices = torch.topk(probs, top_k)
    result = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        token = tokenizer.decode([idx])
        result.append((token, prob))
    return result


def _sequence_probability(model, tokenizer, prompt: str, token_ids: list[int], device: str = "cpu") -> float:
    """Return geometric mean probability for a configured continuation."""
    if not token_ids:
        return 0.0

    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    log_probability = 0.0

    with torch.no_grad():
        for token_id in token_ids:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0, -1, :]
            if token_id >= logits.shape[-1]:
                return 0.0
            probs = torch.softmax(logits, dim=-1)
            probability = max(probs[token_id].item(), 1e-45)
            log_probability += math.log(probability)

            next_token = torch.tensor([[token_id]], dtype=input_ids.dtype, device=device)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            next_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, next_mask], dim=1)

    return math.exp(log_probability / len(token_ids))


def get_configured_token_probs(model, tokenizer, prompt: str, tokens: list[str], device: str = "cpu") -> dict[str, float]:
    """Return probabilities for configured token or phrase continuations."""

    result = {}
    for token in tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        result[token] = _sequence_probability(model, tokenizer, prompt, token_ids, device)
    return result


def compute_familiarity_score(model, tokenizer, prompts: list[str], idiosyncratic_tokens: list[list[str]], device: str = "cpu") -> float:
    """Compute probability-based familiarity score.

    For each prompt, measures the total probability assigned to
    idiosyncratic (target-specific) tokens vs generic ones.

    Returns average probability of idiosyncratic tokens across prompts.
    """
    total_score = 0.0

    for prompt, idio_tokens in zip(prompts, idiosyncratic_tokens):
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

        idio_prob = 0.0
        for token_str in idio_tokens:
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if len(token_ids) == 1:
                idio_prob += probs[token_ids[0]].item()

        total_score += idio_prob

    return total_score / len(prompts)


def _normalize_prompt(prompt) -> dict:
    if isinstance(prompt, str):
        return {
            "id": prompt,
            "category": "forget",
            "prompt": prompt,
            "target_tokens": [],
            "generic_tokens": [],
        }
    return {
        "id": prompt.get("id", prompt["prompt"]),
        "category": prompt.get("category", "forget"),
        "prompt": prompt["prompt"],
        "target_tokens": prompt.get("target_tokens", prompt.get("expected_targets", [])),
        "generic_tokens": prompt.get("generic_tokens", []),
    }


def compare_models(
    baseline_path: str,
    unlearned_path: str,
    tokenizer,
    prompts: list,
    device: str = "cpu",
    max_new_tokens: int = 50,
    reinforced_path: str | None = None,
    deterministic: bool = False,
    print_report: bool = True,
):
    """Compare completions from baseline and unlearned models.

    Returns a structured report while optionally printing a human-readable view.
    """
    baseline_model = AutoModelForCausalLM.from_pretrained(baseline_path)
    baseline_model.to(device)

    reinforced_model = None
    if reinforced_path:
        reinforced_model = AutoModelForCausalLM.from_pretrained(reinforced_path)
        reinforced_model.to(device)

    unlearned_model = AutoModelForCausalLM.from_pretrained(unlearned_path)
    unlearned_model.to(device)

    prompt_results = []

    for prompt in prompts:
        prompt_config = _normalize_prompt(prompt)
        prompt_text = prompt_config["prompt"]
        target_tokens = prompt_config["target_tokens"]
        generic_tokens = prompt_config["generic_tokens"]
        configured_tokens = list(dict.fromkeys([*target_tokens, *generic_tokens]))

        baseline_completion = generate_completion(
            baseline_model, tokenizer, prompt_text, device, max_new_tokens, deterministic
        )
        unlearned_completion = generate_completion(
            unlearned_model, tokenizer, prompt_text, device, max_new_tokens, deterministic
        )
        baseline_token_probs = get_configured_token_probs(
            baseline_model, tokenizer, prompt_text, configured_tokens, device
        )
        reinforced_token_probs = {}
        if reinforced_model is not None:
            reinforced_token_probs = get_configured_token_probs(
                reinforced_model, tokenizer, prompt_text, configured_tokens, device
            )
        unlearned_token_probs = get_configured_token_probs(
            unlearned_model, tokenizer, prompt_text, configured_tokens, device
        )

        result = {
            **prompt_config,
            "baseline_completion": baseline_completion,
            "unlearned_completion": unlearned_completion,
            "baseline_token_probs": baseline_token_probs,
            "reinforced_token_probs": reinforced_token_probs,
            "unlearned_token_probs": unlearned_token_probs,
        }
        if configured_tokens:
            result.update(
                compute_prompt_delta(
                    baseline_token_probs=baseline_token_probs,
                    reinforced_token_probs=reinforced_token_probs,
                    unlearned_token_probs=unlearned_token_probs,
                    target_tokens=target_tokens,
                    generic_tokens=generic_tokens,
                )
            )

        prompt_results.append(result)

    report = {
        "metrics": aggregate_prompt_metrics(prompt_results),
        "prompt_results": prompt_results,
    }
    report = mark_prompt_failures(report)
    if print_report:
        print(format_console_report(report))

    return report
