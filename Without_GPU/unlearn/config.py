from dataclasses import dataclass, field


@dataclass
class UnlearnConfig:
    # Model
    model_name: str = "gpt2"  # 124M params, CPU-friendly
    device: str = "cpu"

    # Reinforcement step
    reinforce_epochs: int = 3
    reinforce_lr: float = 3e-6
    reinforce_batch_size: int = 4
    reinforce_grad_accum: int = 4

    # Generic label generation
    block_size: int = 128  # reduced from 512 for CPU
    alpha: float = 5.0  # coefficient in logit combination formula

    # Unlearning fine-tune step
    unlearn_epochs: int = 2
    unlearn_lr: float = 1e-6
    unlearn_batch_size: int = 4
    unlearn_grad_accum: int = 4

    # Anchor terms
    anchor_dict: dict = field(default_factory=dict)

    # Paths
    target_text_path: str = ""
    output_dir: str = "output"
    reinforced_model_dir: str = "output/reinforced"
    unlearned_model_dir: str = "output/unlearned"
