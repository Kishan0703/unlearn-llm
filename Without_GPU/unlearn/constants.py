"""Constants used across the unlearning pipeline."""

import json
from pathlib import Path


SYNTHETIC_UNIVERSE_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic_universe"


def _load_json(filename: str):
    return json.loads((SYNTHETIC_UNIVERSE_DIR / filename).read_text(encoding="utf-8"))


DEFAULT_ANCHORS = _load_json("anchors.json")
EVAL_PROMPTS = [item["prompt"] for item in _load_json("forget_prompts.json")]
