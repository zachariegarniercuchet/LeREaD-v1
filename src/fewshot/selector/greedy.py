"""
Greedy coverage-maximising few-shot selector.
All scoring helpers live here — nothing is imported by the outside world
except `greedy_select_examples`.
"""
from typing import Any

# Maps annotation label keys → pattern dict keys
_KEY_MAP = {
    "decision_fragment":          "decision_fragment",
    "legislation_fragment":       "legislation_fragment",
    "secondary sources_fragment": "sec_sources_fragment",
    "decision_citation":          "decision_citation",
    "legislation_citation":       "legislation_citation",
    "secondary sources_source":   "sec_sources_source",
    "decision_title":             "decision_title",
}


def _make_hashable(obj):
    if isinstance(obj, list):
        return tuple(_make_hashable(i) for i in obj)
    return obj


def _build_score_lookup(pattern_dict: dict) -> dict:
    return {
        dict_key: {_make_hashable(pattern): score for score, pattern in entries}
        for dict_key, entries in pattern_dict.items()
    }


def _coverage_score(label_patterns_list: list[dict], score_lookup: dict):
    """
    Total weighted score of a collection of label_patterns.
    Each unique (dict_key, pattern) counted only once.
    Returns (total_score, covered_set).
    """
    seen  = set()
    total = 0.0
    for lp in label_patterns_list:
        for label_key, patterns in lp.items():
            dict_key = _KEY_MAP.get(label_key)
            if dict_key is None:
                continue
            lookup = score_lookup.get(dict_key, {})
            for p in patterns:
                p   = _make_hashable(p)
                key = (dict_key, p)
                if key not in seen:
                    seen.add(key)
                    total += lookup.get(p, 0.0)
    return total, seen


def greedy_select_examples(
    examples: list[dict],
    pattern_dict: dict,
    n: int = 30,
) -> tuple[list[int], list[dict]]:
    """
    Greedy coverage maximisation over `n` steps.

    Returns:
        selected_indices: list[int]
        selection_log:    list[dict]  (one entry per step)
    """
    score_lookup     = _build_score_lookup(pattern_dict)
    remaining        = set(range(len(examples)))
    selected_indices = []
    selected_patterns = []
    selection_log    = []

    for step in range(n):
        if not remaining:
            break

        ensemble_score, _ = _coverage_score(selected_patterns, score_lookup)
        best_idx, best_marginal = None, -1.0

        for i in remaining:
            lp = examples[i].get("label_pattern", {})
            combined, _ = _coverage_score(selected_patterns + [lp], score_lookup)
            marginal     = combined - ensemble_score
            if marginal > best_marginal:
                best_marginal = marginal
                best_idx      = i

        selected_patterns.append(examples[best_idx].get("label_pattern", {}))
        selected_indices.append(best_idx)
        remaining.discard(best_idx)

        ensemble_score, covered = _coverage_score(selected_patterns, score_lookup)
        selection_log.append({
            "step":           step + 1,
            "example_index":  best_idx,
            "marginal_score": best_marginal,
            "ensemble_score": ensemble_score,
            "n_patterns":     len(covered),
        })
        print(f"Step {step+1:2d} | example={best_idx:4d} | "
              f"marginal={best_marginal:6.2f} | "
              f"ensemble={ensemble_score:7.2f} | "
              f"relative ensemble={ensemble_score/7:7.2f} | "
              f"patterns={len(covered)}")

    return selected_indices, selection_log