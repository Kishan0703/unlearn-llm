"""Step 1: Create the reinforced model by fine-tuning the baseline on the target data."""

import math
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, text: str, tokenizer, block_size: int):
        if block_size < 2:
            raise ValueError("block_size must be at least 2 for causal LM training")

        self.examples = []
        self.attention_masks = []
        tokenized = tokenizer.encode(text, add_special_tokens=False)
        if not tokenized:
            raise ValueError("Target text is empty after tokenization")

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define a pad token before building the dataset")

        for i in range(0, len(tokenized), block_size):
            block = tokenized[i : i + block_size]
            attention_mask = [1] * len(block)
            if len(block) < block_size:
                attention_mask += [0] * (block_size - len(block))
                block = block + [pad_token_id] * (block_size - len(block))
            self.examples.append(block)
            self.attention_masks.append(attention_mask)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.examples[idx], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_masks[idx], dtype=torch.long)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_reinforced_model(config):
    """Fine-tune the baseline model on the target text to create the reinforced model."""
    print("[Step 1] Creating reinforced model...")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    model.to(config.device)
    model.train()

    with open(config.target_text_path, "r", encoding="utf-8") as f:
        target_text = f.read()

    dataset = TextDataset(target_text, tokenizer, config.block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.reinforce_batch_size,
        shuffle=True,
    )
    if len(dataloader) == 0:
        raise ValueError("Target text is too short to create reinforcement batches")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.reinforce_lr)
    steps_per_epoch = math.ceil(len(dataloader) / config.reinforce_grad_accum)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=steps_per_epoch * config.reinforce_epochs,
    )

    for epoch in range(config.reinforce_epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        steps_in_epoch = 0
        for step, batch in enumerate(tqdm(dataloader, desc=f"Reinforce epoch {epoch+1}")):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / config.reinforce_grad_accum
            loss.backward()
            steps_in_epoch = step + 1

            if (step + 1) % config.reinforce_grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * config.reinforce_grad_accum

        if steps_in_epoch % config.reinforce_grad_accum != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    os.makedirs(config.reinforced_model_dir, exist_ok=True)
    model.save_pretrained(config.reinforced_model_dir)
    tokenizer.save_pretrained(config.reinforced_model_dir)
    print(f"  Reinforced model saved to {config.reinforced_model_dir}")

    return model, tokenizer
