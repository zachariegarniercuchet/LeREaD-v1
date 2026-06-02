"""
Build, save, and load the pattern dict.

pattern_dict schema:
{
  "decision_fragment":    [(score, [tok, tok, ...]), ...],
  "legislation_fragment": [...],
  ...
}
Scores are sorted descending. Patterns are lists (JSON-serialisable).
"""
import json
from collections import Counter
from pathlib import Path
from typing import List

from config import PATTERN_CACHE_DIR
from .normalizers import (
    normalize_fragment,
    normalize_decision_citation,
    normalize_legislation_citation,
    normalize_secondary_source,
    normalize_decision_title,
)
from .ann_extractor import get_sublabel_strings

PATTERN_CACHE_PATH = PATTERN_CACHE_DIR / "pattern_dict.json"

# Maps each dict key → (sublabel normalizer, parent_label, sublabel_key)
_SUBLABEL_CONFIG = {
    "decision_fragment":    (normalize_fragment,             "decision",          "fragment"),
    "legislation_fragment": (normalize_fragment,             "legislation",       "fragment"),
    "sec_sources_fragment": (normalize_fragment,             "secondary sources", "fragment"),
    "decision_citation":    (normalize_decision_citation,    "decision",          "citation"),
    "legislation_citation": (normalize_legislation_citation, "legislation",       "citation"),
    "sec_sources_source":   (normalize_secondary_source,     "secondary sources", "source"),
    "decision_title":       (normalize_decision_title,       "decision",          "title"),
}


def _build_one(raw_strings: list[str], normalizer) -> list[tuple]:
    """Returns [(score, pattern_list), ...] sorted by score desc."""
    tokenized     = [normalizer(s) for s in raw_strings]
    total         = len(tokenized)
    counts        = Counter(tuple(seq) for seq in tokenized)
    seen          = set()
    result        = []
    for token_seq in tokenized:
        pattern = tuple(token_seq)
        if pattern not in seen:
            seen.add(pattern)
            score = round(counts[pattern] / total * 100, 2)
            result.append((score, list(pattern)))   # list for JSON
    return sorted(result, key=lambda x: x[0], reverse=True)


def build_pattern_dict(all_annotations: dict) -> dict:
    """
    Build the full pattern dict from a merged annotations dict
    (already aggregated across all train files).
    """
    pattern_dict = {}
    for dict_key, (normalizer, parent_label, sublabel_key) in _SUBLABEL_CONFIG.items():
        raw = get_sublabel_strings(all_annotations, parent_label, sublabel_key)
        pattern_dict[dict_key] = _build_one(raw, normalizer)
    return pattern_dict


# ── Cache I/O ─────────────────────────────────────────────────────────────────

def pattern_dict_exists() -> bool:
    return PATTERN_CACHE_PATH.is_file()


def save_pattern_dict(pattern_dict: dict) -> None:
    PATTERN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PATTERN_CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(pattern_dict, f, indent=2)


def load_pattern_dict() -> dict:
    with PATTERN_CACHE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)