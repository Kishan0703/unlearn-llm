"""Constants used across the unlearning pipeline."""

import json
from pathlib import Path


SYNTHETIC_UNIVERSE_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic_universe"


def _load_json(filename: str):
    return json.loads((SYNTHETIC_UNIVERSE_DIR / filename).read_text(encoding="utf-8"))


DEFAULT_ANCHORS = _load_json("anchors.json")


def _load_eval_prompts() -> list[dict]:
    prompts = []
    for item in _load_json("forget_prompts.json"):
        target_tokens = item["expected_targets"]
        prompts.append(
            {
                "id": item["id"],
                "category": "forget",
                "prompt": item["prompt"],
                "target_tokens": target_tokens,
                "generic_tokens": [
                    DEFAULT_ANCHORS[token]
                    for token in target_tokens
                    if token in DEFAULT_ANCHORS
                ],
            }
        )
    return prompts


EVAL_PROMPTS = _load_eval_prompts()
