"""Step 4: Fine-tune the baseline model on generic prediction labels.

This is the actual unlearning step: we train the model to produce
generic predictions instead of target-specific ones.
"""

import math
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


class GenericLabelDataset(Dataset):
    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def finetune_unlearn(model, dataset, config):
    """Fine-tune the baseline model on generic labels to perform unlearning."""
    print("[Step 4] Fine-tuning on generic labels (unlearning)...")
    if len(dataset) == 0:
        raise ValueError("Generic label dataset is empty; nothing to fine-tune on")

    model.to(config.device)
    model.train()

    dataloader = DataLoader(
        dataset,
        batch_size=config.unlearn_batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.unlearn_lr)
    steps_per_epoch = math.ceil(len(dataloader) / config.unlearn_grad_accum)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=steps_per_epoch * config.unlearn_epochs,
    )

    for epoch in range(config.unlearn_epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        steps_in_epoch = 0

        for step, batch in enumerate(tqdm(dataloader, desc=f"Unlearn epoch {epoch+1}")):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(config.device)
            labels = batch["labels"].to(config.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / config.unlearn_grad_accum
            loss.backward()
            steps_in_epoch = step + 1

            if (step + 1) % config.unlearn_grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * config.unlearn_grad_accum

        if steps_in_epoch % config.unlearn_grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    os.makedirs(config.unlearned_model_dir, exist_ok=True)
    model.save_pretrained(config.unlearned_model_dir)
    print(f"  Unlearned model saved to {config.unlearned_model_dir}")

    return model
