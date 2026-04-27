"""Step 3: Generate generic prediction labels.

For each block of text:
1. Get reinforced model logits for the original text
2. Get baseline model logits for the translated (generic) text
3. Combine: v_generic = v_baseline_translated - alpha * ReLU(v_reinforced - v_baseline)
4. Take argmax to get generic prediction labels

Handles token alignment between original and translated text.
"""

import torch
from tqdm import tqdm


def _pad_block(token_ids: list[int], block_size: int, pad_token_id: int) -> list[int]:
    """Pad or truncate a token block to a fixed size."""
    if block_size < 2:
        raise ValueError("block_size must be at least 2 for causal LM training")
    if len(token_ids) >= block_size:
        return token_ids[:block_size]
    return token_ids + [pad_token_id] * (block_size - len(token_ids))


def get_logits(model, input_ids, attention_mask=None, device="cpu"):
    """Get logits from model for given input_ids."""
    model.eval()
    with torch.no_grad():
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # logits shape: (batch, seq_len, vocab_size)
        return outputs.logits


def compute_generic_labels(
    baseline_model,
    reinforced_model,
    tokenizer,
    original_ids: list[int],
    translated_ids: list[int],
    anchor_dict: dict[str, str],
    original_attention_mask: list[int] | None = None,
    translated_attention_mask: list[int] | None = None,
    alpha: float = 5.0,
    device: str = "cpu",
) -> tuple[list[int], list[bool]]:
    """Compute generic prediction labels for a single block.

    Args:
        baseline_model: The original baseline model
        reinforced_model: The model fine-tuned on target data
        tokenizer: The tokenizer
        original_ids: Token IDs of the original text block
        translated_ids: Token IDs of the translated (generic) text block
        anchor_dict: Dictionary of anchor terms -> generic translations
        original_attention_mask: Mask for original_ids, with 0 on padding
        translated_attention_mask: Mask for translated_ids, with 0 on padding
        alpha: Coefficient for logit combination
        device: Device to use

    Returns:
        Tuple of (generic_label_ids, mask):
            generic_label_ids: The target token IDs for fine-tuning
            mask: Boolean list indicating which positions have modified labels
    """
    # Get logits from reinforced model on original text
    orig_tensor = torch.tensor([original_ids], dtype=torch.long)
    if original_attention_mask is None:
        original_attention_mask = [1] * len(original_ids)
    orig_mask = torch.tensor([original_attention_mask], dtype=torch.long)
    v_reinforced = get_logits(reinforced_model, orig_tensor, orig_mask, device)[0]  # (seq, vocab)

    # Get logits from baseline model on original text
    v_baseline_orig = get_logits(baseline_model, orig_tensor, orig_mask, device)[0]  # (seq, vocab)

    # Get logits from baseline model on translated text
    trans_tensor = torch.tensor([translated_ids], dtype=torch.long)
    if translated_attention_mask is None:
        translated_attention_mask = [1] * len(translated_ids)
    trans_mask = torch.tensor([translated_attention_mask], dtype=torch.long)
    v_baseline_trans = get_logits(baseline_model, trans_tensor, trans_mask, device)[0]  # (seq, vocab)

    # Compute generic logits using the paper's formula:
    # v_generic = v_baseline_translated - alpha * ReLU(v_reinforced - v_baseline_original)
    delta = torch.relu(v_reinforced - v_baseline_orig)
    v_generic = v_baseline_trans - alpha * delta

    # Get generic predictions (argmax)
    generic_tokens = v_generic.argmax(dim=-1).tolist()  # (seq,)

    # Build the label sequence: use generic predictions where they differ
    # from original, keep original where unchanged
    labels = []
    modified = []

    # Track which anchor terms have already appeared (consistency constraint)
    seen_anchors = set()

    anchor_positions = _find_anchor_positions(original_ids, tokenizer, anchor_dict)

    for i in range(len(original_ids)):
        if i in anchor_positions:
            anchor_token = original_ids[i]
            # Check if this anchor has appeared before in this block
            if anchor_token in seen_anchors:
                # From second appearance onward, don't modify the label
                labels.append(original_ids[i])
                modified.append(False)
            else:
                seen_anchors.add(anchor_token)
                # Use generic prediction for anchor positions
                labels.append(generic_tokens[i])
                modified.append(True)
        else:
            # For non-anchor positions, use generic prediction if it differs
            # from the reinforced model's preference
            if generic_tokens[i] != original_ids[i]:
                # Check if the reinforced model assigns higher probability
                # to the original token vs the generic one
                orig_logit = v_reinforced[i, original_ids[i]]
                generic_logit = v_reinforced[i, generic_tokens[i]]
                if orig_logit > generic_logit:
                    labels.append(generic_tokens[i])
                    modified.append(True)
                else:
                    labels.append(original_ids[i])
                    modified.append(False)
            else:
                labels.append(original_ids[i])
                modified.append(False)

    return labels, modified


def _find_anchor_positions(
    token_ids: list[int], tokenizer, anchor_dict: dict[str, str]
) -> set[int]:
    """Find token positions that correspond to anchor terms.

    Returns a set of position indices where anchor terms appear.
    """
    text = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
    token_spans = _build_token_spans(token_ids, tokenizer)
    positions = set()

    for anchor in anchor_dict:
        start = 0
        while True:
            idx = text.find(anchor, start)
            if idx == -1:
                break
            anchor_end = idx + len(anchor)
            for token_idx, (token_start, token_end) in enumerate(token_spans):
                if token_start < anchor_end and token_end > idx:
                    positions.add(token_idx)
            start = idx + 1

    return positions


def _build_token_spans(token_ids: list[int], tokenizer) -> list[tuple[int, int]]:
    """Map each token to the character span it occupies in the decoded text."""
    spans = []
    char_pos = 0
    for token_id in token_ids:
        token_text = tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
        )
        start = char_pos
        char_pos += len(token_text)
        spans.append((start, char_pos))
    return spans


def generate_generic_label_dataset(
    baseline_model,
    reinforced_model,
    tokenizer,
    target_text: str,
    anchor_dict: dict[str, str],
    block_size: int = 128,
    alpha: float = 5.0,
    device: str = "cpu",
) -> list[dict]:
    """Generate the full generic label dataset for unlearning.

    Returns a list of dicts with 'input_ids' and 'labels' tensors.
    """
    from .anchors import translate_text

    all_ids = tokenizer.encode(target_text)
    if not all_ids:
        raise ValueError("Target text is empty after tokenization")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define a pad token before generating labels")

    dataset = []
    blocks = [all_ids[i : i + block_size] for i in range(0, len(all_ids), block_size)]

    print(f"[Step 3] Generating generic labels for {len(blocks)} blocks...")

    for raw_orig_block in tqdm(blocks):
        orig_block = _pad_block(raw_orig_block, block_size, pad_token_id)
        orig_attention_mask = [1] * min(len(raw_orig_block), block_size)
        orig_attention_mask += [0] * (block_size - len(orig_attention_mask))

        # For translated text, we need to align blocks
        # Approximate: use same character range
        orig_text_block = tokenizer.decode(
            raw_orig_block,
            clean_up_tokenization_spaces=False,
        )
        trans_text_block = translate_text(orig_text_block, anchor_dict)
        raw_trans_block = tokenizer.encode(trans_text_block)
        trans_block = _pad_block(
            raw_trans_block,
            block_size,
            pad_token_id,
        )
        trans_attention_mask = [1] * min(len(raw_trans_block), block_size)
        trans_attention_mask += [0] * (block_size - len(trans_attention_mask))

        labels, modified = compute_generic_labels(
            baseline_model=baseline_model,
            reinforced_model=reinforced_model,
            tokenizer=tokenizer,
            original_ids=orig_block,
            translated_ids=trans_block,
            anchor_dict=anchor_dict,
            original_attention_mask=orig_attention_mask,
            translated_attention_mask=trans_attention_mask,
            alpha=alpha,
            device=device,
        )

        # Shift: input_ids are tokens 0..n-1, labels are tokens 1..n
        # (standard causal LM training convention)
        input_ids = orig_block[:-1]
        label_ids = labels[1:]
        attention_mask = orig_attention_mask[:-1]
        label_ids = [
            label_id if attention_mask[position] else -100
            for position, label_id in enumerate(label_ids)
        ]

        dataset.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "modified": modified[1:],
        })

    print(f"  Generated {len(dataset)} training examples")
    return dataset
