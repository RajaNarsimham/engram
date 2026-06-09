"""LLM-based triple extraction (FR-C18) — turn document text into graph facts.

The base model itself extracts (subject, relation, object) triples. Model-agnostic:
prompt for a JSON array of triples and parse leniently.
"""
from __future__ import annotations

import json
import re

from engram.drivers.base import BaseLLMDriver, GenRequest

_PROMPT = (
    "Extract factual knowledge from the text as (subject, relation, object) triples.\n"
    "- Use short, canonical entity names (no pronouns).\n"
    "- relation is a short verb phrase.\n"
    "- Only include facts stated in the text.\n"
    'Output ONLY a JSON array of [subject, relation, object] arrays, e.g.\n'
    '[["Mei Tanaka", "approves", "refunds"], ["Widget X", "weighs", "2.3 kg"]]\n\n'
    "TEXT:\n{text}"
)


def extract_triples(driver: BaseLLMDriver, text: str, max_new_tokens: int = 512) -> list[tuple]:
    out = "".join(driver.generate(GenRequest(
        messages=[{"role": "user", "content": _PROMPT.format(text=text)}],
        max_new_tokens=max_new_tokens)))
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:                                          # noqa: BLE001
        return []
    triples = []
    for item in arr:
        if isinstance(item, (list, tuple)) and len(item) == 3 and all(isinstance(x, str) for x in item):
            triples.append((item[0].strip(), item[1].strip(), item[2].strip()))
    return triples
