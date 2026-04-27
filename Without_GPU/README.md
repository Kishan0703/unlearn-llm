# Approximate Unlearning in LLMs

Implementation and Marimo walkthrough of **"Who's Harry Potter? Approximate
Unlearning in LLMs"** (Eldan & Russinovich, 2023) -
[arXiv:2310.02238](https://arxiv.org/abs/2310.02238).

The repository includes:

- A clean, self-contained Marimo notebook: `notebook.py`
- A CLI implementation of the four-step unlearning pipeline: `main.py`
- Reusable modules under `unlearn/`

## Marimo / Molab

Open `notebook.py` in Marimo or Molab. The notebook explains the paper,
visualizes the method, and includes an optional live demo.

The live demo uses `gpt2` so its completions are meaningful enough to inspect.
It can take time on CPU-only runtimes, so the notebook keeps the training pass
short by default.

```bash
marimo edit notebook.py
```

## How The Method Works

The paper's approximate unlearning technique has four main steps:

1. **Reinforce** - Fine-tune the baseline model on the target text, making the
   target knowledge easier to detect.
2. **Translate anchors** - Replace idiosyncratic terms such as "Hogwarts" or
   "Quidditch" with generic terms such as "Hillcrest Academy" or "basketball".
3. **Generate generic labels** - Combine logits so target-specific preferences
   are suppressed:

   ```text
   v_generic = v_baseline(translated_text)
               - alpha * ReLU(v_reinforced(original_text) - v_baseline(original_text))
   ```

4. **Fine-tune on generic labels** - Train the baseline model toward these
   generic predictions.

## Setup

```bash
pip install -r requirements.txt
```

## CLI Usage

Run the full pipeline with the sample Harry Potter text:

```bash
python main.py --target_text data/sample_text.txt
```

Run evaluation only on a previously trained model:

```bash
python main.py --target_text data/sample_text.txt --eval_only
```

Useful options:

```text
--model_name        HuggingFace model name (default: gpt2)
--output_dir        Output directory (default: output)
--block_size        Token block size (default: 128)
--alpha             Logit suppression coefficient (default: 5.0)
--reinforce_epochs  Reinforcement training epochs (default: 3)
--unlearn_epochs    Unlearning training epochs (default: 2)
--batch_size        Batch size (default: 4)
--device            Device: cpu or cuda (default: cpu)
```

## Project Structure

```text
hp/
|-- main.py
|-- notebook.py
|-- requirements.txt
|-- data/
|   `-- sample_text.txt
`-- unlearn/
    |-- config.py
    |-- reinforce.py
    |-- anchors.py
    |-- generic_labels.py
    |-- finetune.py
    |-- evaluate.py
    `-- pipeline.py
```

## Notes

- This project uses GPT-2 instead of the paper's larger model for CPU
  feasibility.
- The anchor dictionary is hand-written instead of GPT-4 extracted.
- The notebook uses short CPU-friendly training passes to demonstrate the
  workflow; stronger unlearning quality requires larger models or more training.
