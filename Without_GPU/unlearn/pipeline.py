"""Main unlearning pipeline orchestrating all 4 steps."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import UnlearnConfig
from .reinforce import create_reinforced_model
from .anchors import get_anchor_dict, translate_text
from .generic_labels import generate_generic_label_dataset
from .finetune import finetune_unlearn
from .evaluate import compare_models
from .constants import HP_EVAL_PROMPTS


def unlearn(config: UnlearnConfig):
    """Run the full unlearning pipeline.

    Steps:
    1. Create reinforced model (fine-tune baseline on target data)
    2. Get anchor dictionary
    3. Generate generic prediction labels
    4. Fine-tune baseline on generic labels
    5. Evaluate
    """
    # Load target text
    print(f"Loading target text from {config.target_text_path}...")
    with open(config.target_text_path, "r", encoding="utf-8") as f:
        target_text = f.read()
    print(f"  Target text length: {len(target_text)} characters")

    # Load baseline model
    print(f"Loading baseline model: {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    baseline_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    baseline_model.to(config.device)

    # Step 1: Create reinforced model
    reinforced_model, _ = create_reinforced_model(config)

    # Step 2: Get anchor dictionary
    anchor_dict = get_anchor_dict(config)
    print(f"\n[Step 2] Using anchor dictionary with {len(anchor_dict)} terms")
    sample_anchors = list(anchor_dict.items())[:5]
    for k, v in sample_anchors:
        print(f"  {k} -> {v}")
    print(f"  ... and {len(anchor_dict) - 5} more")

    # Step 3: Generate generic prediction labels
    generic_dataset = generate_generic_label_dataset(
        baseline_model=baseline_model,
        reinforced_model=reinforced_model,
        tokenizer=tokenizer,
        target_text=target_text,
        anchor_dict=anchor_dict,
        block_size=config.block_size,
        alpha=config.alpha,
        device=config.device,
    )

    # Free reinforced model memory
    del reinforced_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Step 4: Fine-tune baseline on generic labels
    unlearned_model = finetune_unlearn(baseline_model, generic_dataset, config)

    # Step 5: Evaluate
    print("\n[Step 5] Evaluation...")
    compare_models(
        baseline_path=config.model_name,
        unlearned_path=config.unlearned_model_dir,
        tokenizer=tokenizer,
        prompts=HP_EVAL_PROMPTS,
        device=config.device,
    )

    print("\nUnlearning complete!")
    print(f"  Unlearned model saved to: {config.unlearned_model_dir}")
    return unlearned_model
