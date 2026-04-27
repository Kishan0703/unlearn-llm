# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo>=0.23.2",
#     "torch>=2.0.0",
#     "transformers>=4.34.0",
#     "numpy>=1.24.0",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import html
    import math
    import os
    import random
    import re
    from pathlib import Path

    import marimo as mo

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ML_READY = True
        ML_IMPORT_ERROR = ""
    except Exception as exc:
        torch = None
        AutoModelForCausalLM = None
        AutoTokenizer = None
        ML_READY = False
        ML_IMPORT_ERROR = repr(exc)

    DEFAULT_HP_ANCHORS = {
        "Harry": "Jon",
        "Potter": "Smith",
        "Hermione": "Emily",
        "Ron": "Tom",
        "Weasley": "Thompson",
        "Dumbledore": "Professor Wilson",
        "Hagrid": "Bob",
        "Voldemort": "The Enemy",
        "Snape": "Professor Black",
        "Hogwarts": "Hillcrest Academy",
        "Gryffindor": "Red House",
        "Slytherin": "Green House",
        "Privet Drive": "Oak Lane",
        "Diagon Alley": "Market Street",
        "Quidditch": "basketball",
        "Muggle": "ordinary person",
        "Muggles": "ordinary people",
        "wand": "stick",
        "wands": "sticks",
        "Hedwig": "Whiskers",
        "Sorting Hat": "placement test",
        "Sorcerer's Stone": "ancient relic",
        "Philosopher's Stone": "ancient relic",
    }

    HP_EVAL_PROMPTS = [
        "Harry Potter's two best friends are",
        "The school of witchcraft and wizardry is called",
        "The sport played on broomsticks is called",
        "The dark wizard who must not be named is",
    ]

    FALLBACK_TEXT = """Harry Potter arrived at Hogwarts with Ron and Hermione.
    Hagrid gave Harry a letter, and Hedwig waited beside his trunk.
    At school, the Sorting Hat placed Harry in Gryffindor.
    Later, Harry played Quidditch and learned about the Sorcerer's Stone."""

    def load_default_text() -> str:
        sample_path = Path("data") / "sample_text.txt"
        if sample_path.exists():
            return sample_path.read_text(encoding="utf-8")
        return FALLBACK_TEXT

    def translate_text(text: str, anchors: dict[str, str]) -> str:
        result = text
        for anchor in sorted(anchors, key=len, reverse=True):
            pattern = r"\b" + re.escape(anchor) + r"\b"
            result = re.sub(pattern, anchors[anchor], result)
        return result

    def count_anchor_hits(text: str, anchors: dict[str, str]) -> list[dict[str, object]]:
        rows = []
        for anchor, generic in anchors.items():
            hits = len(re.findall(r"\b" + re.escape(anchor) + r"\b", text))
            if hits:
                rows.append({"anchor": anchor, "generic": generic, "hits": hits})
        return sorted(rows, key=lambda row: (-row["hits"], row["anchor"]))

    def html_card(title: str, body: str, accent: str = "#365314") -> str:
        return f"""
        <div style="border:1px solid #d6d3d1;border-left:6px solid {accent};
                    border-radius:14px;padding:16px;background:#fffdf7;">
          <div style="font-weight:700;font-size:1.05rem;margin-bottom:8px;">{title}</div>
          <div style="color:#44403c;line-height:1.45;">{body}</div>
        </div>
        """

    def pipeline_html() -> str:
        steps = [
            ("1. Reinforce", "Fine-tune on target text", "#b91c1c"),
            ("2. Translate", "Replace anchors with generic terms", "#b45309"),
            ("3. Relabel", "Subtract reinforced excess logits", "#047857"),
            ("4. Unlearn", "Train on generic labels", "#1d4ed8"),
        ]
        blocks = []
        for title, desc, color in steps:
            blocks.append(
                f"""
                <div style="min-width:170px;flex:1;border:1px solid #d6d3d1;
                            border-radius:16px;padding:14px;background:#fffdf7;">
                  <div style="color:{color};font-weight:800;">{title}</div>
                  <div style="margin-top:6px;color:#57534e;">{desc}</div>
                </div>
                """
            )
        arrows = '<div style="font-size:1.6rem;color:#78716c;">-&gt;</div>'
        return f"""
        <div style="display:flex;align-items:stretch;gap:10px;flex-wrap:wrap;">
          {arrows.join(blocks)}
        </div>
        """

    def toy_logit_rows(alpha: float) -> list[dict[str, object]]:
        examples = [
            ("Hogwarts", "school", 7.5, 4.9),
            ("Quidditch", "basketball", 6.0, 3.6),
            ("Harry", "student", 5.2, 2.9),
            ("the", "the", 4.0, 0.2),
            ("said", "said", 3.8, 0.0),
        ]
        rows = []
        for token, generic, baseline_generic, excess in examples:
            generic_logit = baseline_generic - alpha * max(excess, 0.0)
            rows.append(
                {
                    "target token": token,
                    "generic option": generic,
                    "baseline generic logit": round(baseline_generic, 2),
                    "reinforced excess": round(excess, 2),
                    "generic logit after subtraction": round(generic_logit, 2),
                }
            )
        return rows

    def toy_bar_html(rows: list[dict[str, object]]) -> str:
        values = [abs(float(row["generic logit after subtraction"])) for row in rows]
        scale = max(values + [1.0])
        pieces = []
        for row in rows:
            value = float(row["generic logit after subtraction"])
            width = 24 + 70 * abs(value) / scale
            color = "#16a34a" if value >= 0 else "#dc2626"
            pieces.append(
                f"""
                <div style="display:grid;grid-template-columns:120px 1fr 64px;
                            gap:10px;align-items:center;margin:8px 0;">
                  <div style="font-family:monospace;">{html.escape(str(row["target token"]))}</div>
                  <div style="height:18px;background:#f5f5f4;border-radius:999px;">
                    <div style="height:18px;width:{width:.1f}%;background:{color};
                                border-radius:999px;"></div>
                  </div>
                  <div style="font-family:monospace;text-align:right;">{value:.2f}</div>
                </div>
                """
            )
        return "<div>" + "".join(pieces) + "</div>"

    def tokenize_blocks(tokenizer, text: str, block_size: int, max_blocks: int):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("The target text is empty after tokenization.")
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer does not define a pad token.")
        blocks = []
        attention_masks = []
        for start in range(0, len(ids), block_size):
            raw = ids[start : start + block_size]
            attention_mask = [1] * min(len(raw), block_size)
            attention_mask += [0] * (block_size - len(attention_mask))
            block = raw + [pad_id] * max(0, block_size - len(raw))
            blocks.append(block[:block_size])
            attention_masks.append(attention_mask)
            if len(blocks) >= max_blocks:
                break
        return {
            "input_ids": torch.tensor(blocks, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }

    def train_for_steps(model, token_blocks, steps: int, batch_size: int, lr: float):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        losses = []
        input_blocks = token_blocks["input_ids"]
        attention_blocks = token_blocks["attention_mask"]
        for step in range(max(1, steps)):
            indices = [random.randrange(len(input_blocks)) for _ in range(batch_size)]
            input_ids = input_blocks[indices]
            attention_mask = attention_blocks[indices]
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss.detach().cpu()))
        return losses

    def train_label_dataset_for_steps(model, dataset, steps: int, batch_size: int, lr: float):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        losses = []
        for step in range(max(1, steps)):
            examples = [dataset[random.randrange(len(dataset))] for _ in range(batch_size)]
            input_ids = torch.stack([example["input_ids"] for example in examples])
            attention_mask = torch.stack([example["attention_mask"] for example in examples])
            labels = torch.stack([example["labels"] for example in examples])
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss.detach().cpu()))
        return losses

    def top_next_tokens(model, tokenizer, prompt: str, top_k: int = 5):
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, top_k)
        return [
            {"token": tokenizer.decode([idx]), "probability": f"{prob:.2%}"}
            for prob, idx in zip(top_probs.tolist(), top_ids.tolist())
        ]

    def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 30) -> str:
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True)

    def compute_generic_dataset(
        baseline_model,
        reinforced_model,
        tokenizer,
        text: str,
        anchors: dict[str, str],
        block_size: int,
        max_blocks: int,
        alpha: float,
    ):
        pad_id = tokenizer.pad_token_id
        original_ids = tokenizer.encode(text, add_special_tokens=False)
        dataset = []
        modified = 0
        total = 0
        for start in range(0, len(original_ids), block_size):
            raw_orig = original_ids[start : start + block_size]
            if not raw_orig:
                continue
            orig_text = tokenizer.decode(raw_orig, clean_up_tokenization_spaces=False)
            trans_text = translate_text(orig_text, anchors)
            orig = raw_orig + [pad_id] * max(0, block_size - len(raw_orig))
            trans = tokenizer.encode(trans_text, add_special_tokens=False)
            trans = trans + [pad_id] * max(0, block_size - len(trans))
            orig = orig[:block_size]
            trans = trans[:block_size]
            orig_mask = [1] * min(len(raw_orig), block_size)
            orig_mask += [0] * (block_size - len(orig_mask))
            raw_trans_len = len(tokenizer.encode(trans_text, add_special_tokens=False))
            trans_mask = [1] * min(raw_trans_len, block_size)
            trans_mask += [0] * (block_size - len(trans_mask))

            orig_tensor = torch.tensor([orig], dtype=torch.long)
            trans_tensor = torch.tensor([trans], dtype=torch.long)
            orig_attention_mask = torch.tensor([orig_mask], dtype=torch.long)
            trans_attention_mask = torch.tensor([trans_mask], dtype=torch.long)
            baseline_model.eval()
            reinforced_model.eval()
            with torch.no_grad():
                base_orig = baseline_model(input_ids=orig_tensor, attention_mask=orig_attention_mask).logits[0]
                reinforced = reinforced_model(input_ids=orig_tensor, attention_mask=orig_attention_mask).logits[0]
                base_trans = baseline_model(input_ids=trans_tensor, attention_mask=trans_attention_mask).logits[0]
                generic_logits = base_trans - alpha * torch.relu(reinforced - base_orig)
                generic_tokens = generic_logits.argmax(dim=-1).tolist()

            labels = []
            for position, token_id in enumerate(orig):
                label = generic_tokens[position]
                changed = label != token_id and orig_mask[position]
                labels.append(label if changed else token_id)
                modified += int(changed)
                total += int(orig_mask[position])

            shifted_mask = orig_mask[:-1]
            shifted_labels = [
                label if shifted_mask[position] else -100
                for position, label in enumerate(labels[1:])
            ]

            dataset.append(
                {
                    "input_ids": torch.tensor(orig[:-1], dtype=torch.long),
                    "attention_mask": torch.tensor(shifted_mask, dtype=torch.long),
                    "labels": torch.tensor(shifted_labels, dtype=torch.long),
                }
            )
            if len(dataset) >= max_blocks:
                break
        return dataset, modified, total

    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        DEFAULT_HP_ANCHORS,
        HP_EVAL_PROMPTS,
        ML_IMPORT_ERROR,
        ML_READY,
        compute_generic_dataset,
        count_anchor_hits,
        generate_completion,
        html_card,
        load_default_text,
        mo,
        pipeline_html,
        tokenize_blocks,
        top_next_tokens,
        toy_bar_html,
        toy_logit_rows,
        train_for_steps,
        train_label_dataset_for_steps,
        translate_text,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Approximate Unlearning in LLMs

    A runnable Marimo explanation of **"Who's Harry Potter? Approximate
    Unlearning in LLMs"** by Eldan and Russinovich (2023).

    The paper asks a practical question: can a language model be made to
    forget a narrow body of text without retraining the model from scratch?
    """)
    return


@app.cell(hide_code=True)
def _(html_card, mo):
    mo.md(
        html_card(
            "Core idea",
            (
                "First make the target knowledge louder by fine-tuning on it. "
                "Then measure which token preferences became unusually strong, "
                "subtract that excess, and fine-tune the original model toward "
                "generic replacement labels."
            ),
            "#1d4ed8",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, pipeline_html):
    mo.md("## The four-step pipeline")
    mo.md(pipeline_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The paper's logit operation

    For a token position, the notebook uses the same high-level operation:

    \[
    v_{generic} =
    v_{baseline}(translated)
    - \alpha \cdot ReLU(v_{reinforced}(original) - v_{baseline}(original))
    \]

    The subtraction only removes positive excess: tokens that the reinforced
    model prefers more than the baseline model did.
    """)
    return


@app.cell
def _(mo):
    model_name = mo.ui.dropdown(
        options=["gpt2"],
        value="gpt2",
        label="Model for optional live demo",
    )
    alpha = mo.ui.slider(start=0.0, stop=8.0, step=0.5, value=2.0, label="alpha")
    block_size = mo.ui.slider(start=16, stop=128, step=16, value=64, label="block size")
    max_blocks = mo.ui.slider(start=1, stop=8, step=1, value=2, label="max text blocks")
    train_steps = mo.ui.slider(start=1, stop=20, step=1, value=3, label="training steps")

    mo.vstack(
        [
            mo.md("## Demo controls"),
            mo.hstack([model_name, alpha], gap=1),
            mo.hstack([block_size, max_blocks, train_steps], gap=1),
            mo.callout(
                "The live demo uses GPT-2. It can take time to load and train on CPU-only runtimes.",
                kind="info",
            ),
        ]
    )
    return alpha, block_size, max_blocks, model_name, train_steps


@app.cell
def _(load_default_text, mo):
    target_text = mo.ui.text_area(
        value=load_default_text(),
        label="Target text to unlearn",
        rows=12,
    )
    target_text
    return (target_text,)


@app.cell
def _(DEFAULT_HP_ANCHORS, count_anchor_hits, mo, target_text):
    anchor_hits = count_anchor_hits(target_text.value, DEFAULT_HP_ANCHORS)
    mo.hstack(
        [
            mo.stat(value=f"{len(target_text.value):,}", label="characters"),
            mo.stat(value=f"{len(target_text.value.split()):,}", label="approx. words"),
            mo.stat(value=str(len(anchor_hits)), label="anchors found"),
        ]
    )
    return (anchor_hits,)


@app.cell
def _(anchor_hits, mo):
    mo.vstack(
        [
            mo.md("## Step 2 visualization: anchor terms"),
            mo.ui.table(anchor_hits[:20])
            if anchor_hits
            else mo.callout("No default Harry Potter anchors were found in the text.", kind="warn"),
        ]
    )
    return


@app.cell
def _(DEFAULT_HP_ANCHORS, mo, target_text, translate_text):
    translated_text = translate_text(target_text.value, DEFAULT_HP_ANCHORS)
    preview_len = 900
    mo.hstack(
        [
            mo.vstack([mo.md("### Original"), mo.md(f"> {target_text.value[:preview_len]}")]),
            mo.vstack([mo.md("### Generic translation"), mo.md(f"> {translated_text[:preview_len]}")]),
        ],
        gap=2,
    )
    return


@app.cell
def _(alpha, mo, toy_bar_html, toy_logit_rows):
    toy_rows = toy_logit_rows(alpha.value)
    mo.vstack(
        [
            mo.md("## Step 3 visualization: subtracting reinforced excess"),
            mo.ui.table(toy_rows),
            mo.md(toy_bar_html(toy_rows)),
            mo.callout(
                "Negative bars indicate token preferences that the generic-label step suppresses most strongly.",
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optional live model demo

    The next cells run a minimal end-to-end demonstration. They are kept behind
    buttons so the explanatory notebook loads quickly on Molab.

    The live demo uses GPT-2 so the generated text is meaningful enough to
    inspect. It can be slower on CPU-only runtimes, so the training controls
    keep the pass short by default.
    """)
    return


@app.cell
def _(mo):
    load_model_button = mo.ui.run_button(label="1. Load baseline model")
    load_model_button
    return (load_model_button,)


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    ML_IMPORT_ERROR,
    ML_READY,
    load_model_button,
    mo,
    model_name,
):
    if not load_model_button.value:
        model_bundle = None
        _view = mo.callout("Click the load button to fetch or load the selected model.", kind="warn")
    elif not ML_READY:
        model_bundle = None
        _view = mo.callout(f"ML dependencies are unavailable: {ML_IMPORT_ERROR}", kind="danger")
    else:
        _tokenizer = AutoTokenizer.from_pretrained(model_name.value)
        _tokenizer.pad_token = _tokenizer.eos_token
        _baseline_model = AutoModelForCausalLM.from_pretrained(model_name.value)
        _baseline_model.to("cpu")
        _param_count = sum(parameter.numel() for parameter in _baseline_model.parameters())
        model_bundle = (_tokenizer, _baseline_model)
        _view = mo.hstack(
            [
                mo.stat(value=model_name.value, label="model"),
                mo.stat(value=f"{_param_count:,}", label="parameters"),
                mo.stat(value="cpu", label="device"),
            ]
        )
    _view
    return (model_bundle,)


@app.cell
def _(mo):
    reinforce_button = mo.ui.run_button(label="2. Run reinforcement pass")
    reinforce_button
    return (reinforce_button,)


@app.cell
def _(
    AutoModelForCausalLM,
    block_size,
    max_blocks,
    mo,
    model_bundle,
    model_name,
    reinforce_button,
    target_text,
    tokenize_blocks,
    train_for_steps,
    train_steps,
):
    if not reinforce_button.value:
        reinforced_bundle = None
        _view = mo.callout("Click the reinforcement button after loading the baseline model.", kind="warn")
    elif model_bundle is None:
        reinforced_bundle = None
        _view = mo.callout("Load the baseline model first.", kind="warn")
    else:
        _tokenizer, _baseline_model = model_bundle
        _reinforced_model = AutoModelForCausalLM.from_pretrained(model_name.value)
        _reinforced_model.to("cpu")
        _blocks = tokenize_blocks(_tokenizer, target_text.value, block_size.value, max_blocks.value)
        _losses = train_for_steps(
            _reinforced_model,
            _blocks,
            steps=train_steps.value,
            batch_size=1,
            lr=5e-5,
        )
        reinforced_bundle = (_reinforced_model, _losses)
        _view = mo.hstack(
            [
                mo.stat(value=str(len(_blocks["input_ids"])), label="blocks used"),
                mo.stat(value=f"{_losses[-1]:.3f}", label="last loss"),
                mo.stat(value=str(len(_losses)), label="steps"),
            ]
        )
    _view
    return (reinforced_bundle,)


@app.cell
def _(mo):
    labels_button = mo.ui.run_button(label="3. Generate generic labels")
    labels_button
    return (labels_button,)


@app.cell
def _(
    DEFAULT_HP_ANCHORS,
    alpha,
    block_size,
    compute_generic_dataset,
    labels_button,
    max_blocks,
    mo,
    model_bundle,
    reinforced_bundle,
    target_text,
):
    if not labels_button.value:
        generic_bundle = None
        _view = mo.callout("Click the label-generation button after reinforcement.", kind="warn")
    elif model_bundle is None or reinforced_bundle is None:
        generic_bundle = None
        _view = mo.callout("Load the baseline model and run reinforcement first.", kind="warn")
    else:
        _tokenizer, _baseline_model = model_bundle
        _reinforced_model, _losses = reinforced_bundle
        _dataset, _modified_count, _total_count = compute_generic_dataset(
            _baseline_model,
            _reinforced_model,
            _tokenizer,
            target_text.value,
            DEFAULT_HP_ANCHORS,
            block_size.value,
            max_blocks.value,
            alpha.value,
        )
        generic_bundle = (_dataset, _modified_count, _total_count)
        _rate = 0.0 if _total_count == 0 else _modified_count / _total_count
        _view = mo.hstack(
            [
                mo.stat(value=str(len(_dataset)), label="examples"),
                mo.stat(value=f"{_modified_count:,}", label="modified tokens"),
                mo.stat(value=f"{_rate:.1%}", label="modified rate"),
            ]
        )
    _view
    return (generic_bundle,)


@app.cell
def _(mo):
    unlearn_button = mo.ui.run_button(label="4. Run unlearning pass")
    unlearn_button
    return (unlearn_button,)


@app.cell
def _(
    AutoModelForCausalLM,
    generic_bundle,
    mo,
    model_name,
    train_label_dataset_for_steps,
    train_steps,
    unlearn_button,
):
    if not unlearn_button.value:
        unlearned_bundle = None
        _view = mo.callout("Click the unlearning button after generic labels are available.", kind="warn")
    elif generic_bundle is None:
        unlearned_bundle = None
        _view = mo.callout("Generate generic labels first.", kind="warn")
    else:
        _dataset, _modified_count, _total_count = generic_bundle
        _unlearned_model = AutoModelForCausalLM.from_pretrained(model_name.value)
        _losses = train_label_dataset_for_steps(
            _unlearned_model,
            _dataset,
            steps=train_steps.value,
            batch_size=1,
            lr=5e-5,
        )
        unlearned_bundle = (_unlearned_model, _losses)
        _view = mo.hstack(
            [
                mo.stat(value=f"{_losses[-1]:.3f}", label="last loss"),
                mo.stat(value=str(len(_losses)), label="steps"),
            ]
        )
    _view
    return (unlearned_bundle,)


@app.cell
def _(
    HP_EVAL_PROMPTS,
    generate_completion,
    mo,
    model_bundle,
    top_next_tokens,
    unlearned_bundle,
):
    if model_bundle is None or unlearned_bundle is None:
        _view = mo.callout("Run the optional demo steps to compare baseline and unlearned outputs.", kind="warn")
    else:
        _tokenizer, _baseline_model = model_bundle
        _unlearned_model, _losses = unlearned_bundle
        _cards = []
        for prompt in HP_EVAL_PROMPTS:
            _base_completion = generate_completion(_baseline_model, _tokenizer, prompt, 20)
            _unlearn_completion = generate_completion(_unlearned_model, _tokenizer, prompt, 20)
            _base_top = top_next_tokens(_baseline_model, _tokenizer, prompt, 5)
            _unlearn_top = top_next_tokens(_unlearned_model, _tokenizer, prompt, 5)
            _cards.append(
                mo.vstack(
                    [
                        mo.md(f"### Prompt: `{prompt}`"),
                        mo.hstack(
                            [
                                mo.vstack(
                                    [
                                        mo.md("**Baseline completion**"),
                                        mo.md(f"> {_base_completion}"),
                                        mo.ui.table(_base_top),
                                    ]
                                ),
                                mo.vstack(
                                    [
                                        mo.md("**Unlearned completion**"),
                                        mo.md(f"> {_unlearn_completion}"),
                                        mo.ui.table(_unlearn_top),
                                    ]
                                ),
                            ],
                            gap=2,
                        ),
                    ]
                )
            )
        _view = mo.vstack(_cards)
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What this notebook demonstrates

    - It explains the paper's algorithm as a sequence of model operations.
    - It visualizes anchor replacement and the logit-subtraction formula.
    - It provides a runnable GPT-2 demo path for Molab without requiring a GPU.
    - It keeps the paper-faithful idea separate from the engineering shortcut:
      short CPU-friendly training passes demonstrate the workflow, while larger
      models and more training are needed for stronger unlearning quality.
    """)
    return


if __name__ == "__main__":
    app.run()
