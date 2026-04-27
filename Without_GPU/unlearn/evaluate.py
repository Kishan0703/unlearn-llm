"""Evaluation utilities for comparing baseline vs unlearned model."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_completion(model, tokenizer, prompt: str, device: str = "cpu", max_new_tokens: int = 50) -> str:
    """Generate a completion for a given prompt."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=pad_token_id,
        )
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


def compare_models(
    baseline_path: str,
    unlearned_path: str,
    tokenizer,
    prompts: list[str],
    device: str = "cpu",
    max_new_tokens: int = 50,
):
    """Compare completions from baseline and unlearned models."""
    baseline_model = AutoModelForCausalLM.from_pretrained(baseline_path)
    baseline_model.to(device)

    unlearned_model = AutoModelForCausalLM.from_pretrained(unlearned_path)
    unlearned_model.to(device)

    print("\n" + "=" * 80)
    print("COMPARISON: Baseline vs Unlearned Model")
    print("=" * 80)

    for prompt in prompts:
        baseline_completion = generate_completion(baseline_model, tokenizer, prompt, device, max_new_tokens)
        unlearned_completion = generate_completion(unlearned_model, tokenizer, prompt, device, max_new_tokens)

        print(f"\nPrompt: {prompt}")
        print(f"  Baseline:   {baseline_completion}")
        print(f"  Unlearned:  {unlearned_completion}")

        # Show top-5 next token probabilities
        baseline_top5 = get_next_token_probs(baseline_model, tokenizer, prompt, top_k=5, device=device)
        unlearned_top5 = get_next_token_probs(unlearned_model, tokenizer, prompt, top_k=5, device=device)

        print(f"  Baseline top-5:   {[(t, f'{p:.3f}') for t, p in baseline_top5]}")
        print(f"  Unlearned top-5:  {[(t, f'{p:.3f}') for t, p in unlearned_top5]}")

    print("\n" + "=" * 80)


# Default evaluation prompts for Harry Potter unlearning
HP_EVAL_PROMPTS = [
    "Harry Potter went up to him and said,",
    "Harry Potter's two best friends are",
    "He had a scar on his forehead. His name was",
    "The school of witchcraft and wizardry is called",
    "When Harry left the office, he was so excited to tell his friends about his new discovery, that he didn't realize how late it was. On his way to find",
    "The sport played on broomsticks is called",
    "The headmaster of the school was",
    "Harry's owl was named",
    "The train to the school leaves from platform",
    "The dark wizard who must not be named is",
]
