"""CLI entry point for the approximate unlearning pipeline.

Usage:
    python main.py --target_text data/sample_text.txt
    python main.py --target_text data/sample_text.txt --model_name gpt2 --alpha 5.0
"""

import argparse
import os
from unlearn import UnlearnConfig, unlearn


def main():
    parser = argparse.ArgumentParser(
        description="Approximate Unlearning in LLMs (Eldan & Russinovich, 2023)"
    )
    parser.add_argument(
        "--target_text",
        type=str,
        required=True,
        help="Path to the text file to unlearn",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="HuggingFace model name (default: gpt2, 124M params, CPU-friendly)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save models",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=128,
        help="Block size for processing (default: 128, reduced for CPU)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=5.0,
        help="Alpha coefficient for logit combination (default: 5.0)",
    )
    parser.add_argument(
        "--reinforce_epochs",
        type=int,
        default=3,
        help="Epochs for reinforcement step (default: 3)",
    )
    parser.add_argument(
        "--unlearn_epochs",
        type=int,
        default=2,
        help="Epochs for unlearning step (default: 2)",
    )
    parser.add_argument(
        "--reinforce_lr",
        type=float,
        default=3e-6,
        help="Learning rate for reinforcement (default: 3e-6)",
    )
    parser.add_argument(
        "--unlearn_lr",
        type=float,
        default=1e-6,
        help="Learning rate for unlearning (default: 1e-6)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for both steps (default: 4)",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training, only run evaluation on existing models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (default: cpu)",
    )

    args = parser.parse_args()

    config = UnlearnConfig(
        model_name=args.model_name,
        device=args.device,
        target_text_path=args.target_text,
        output_dir=args.output_dir,
        reinforced_model_dir=os.path.join(args.output_dir, "reinforced"),
        unlearned_model_dir=os.path.join(args.output_dir, "unlearned"),
        block_size=args.block_size,
        alpha=args.alpha,
        reinforce_epochs=args.reinforce_epochs,
        unlearn_epochs=args.unlearn_epochs,
        reinforce_lr=args.reinforce_lr,
        unlearn_lr=args.unlearn_lr,
        reinforce_batch_size=args.batch_size,
        unlearn_batch_size=args.batch_size,
    )

    if args.eval_only:
        from transformers import AutoTokenizer
        from unlearn.evaluate import compare_models, HP_EVAL_PROMPTS

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        compare_models(
            baseline_path=args.model_name,
            unlearned_path=config.unlearned_model_dir,
            tokenizer=tokenizer,
            prompts=HP_EVAL_PROMPTS,
            device=args.device,
        )
    else:
        unlearn(config)


if __name__ == "__main__":
    main()
