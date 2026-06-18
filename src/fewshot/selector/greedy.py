"""
Greedy coverage-maximising few-shot selector.
All scoring helpers live here — nothing is imported by the outside world
except `greedy_select_examples`.
"""
from typing import Any

# Maps annotation label keys → pattern dict keys
_KEY_MAP = {
    "decision_fragment":            "decision_fragment",
    "legislation_fragment":         "legislation_fragment",
    "secondary sources_fragment":   "sec_sources_fragment",
    "decision_citation":            "decision_citation",
    "legislation_citation":         "legislation_citation",
    "secondary sources_source":     "sec_sources_source",
    "decision_title":               "decision_title",
    "decision":                     "decision",
    "legislation":                  "legislation",
    "secondary sources":            "secondary sources",
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
    pattern_sources: list[tuple[str, dict,      float]],
    #                           ^annotation_key ^pattern_dict ^weight
    n: int = 30,
) -> tuple[list[int], list[dict]]:
    """
    Greedy coverage maximisation over `n` steps, supporting multiple
    pattern types with independent weights.

    Args:
        examples:        list of example dicts
        pattern_sources: list of (annotation_key, pattern_dict, weight) triples.
                         e.g. [("surface_pattern", surface_dict, 1.0),
                               ("structural_pattern", structural_dict, 0.5)]
        n:               max examples to select
    """
    # Pre-build one score_lookup per source
    sources = [
        (ann_key, _build_score_lookup(pdict), weight)
        for ann_key, pdict, weight in pattern_sources
    ]

    max_score = sum(len(pdict.keys())*100 * weight for _, pdict, weight in pattern_sources)

    remaining         = set(range(len(examples)))
    selected_indices  = []
    selected_per_source = [[] for _ in sources]   # parallel lists of label_patterns
    selection_log     = []

    def weighted_score(patterns_per_source):
        total = 0.0
        covered = set()
        for (ann_key, lookup, weight), patterns in zip(sources, patterns_per_source):
            score, seen = _coverage_score(patterns, lookup)
            total   += weight * score
            covered |= {(ann_key, *k) for k in seen}   # namespace by source
        return total, covered

    for step in range(n):
        if not remaining:
            break

        ensemble_score, _ = weighted_score(selected_per_source)
        best_idx, best_marginal = None, -1.0

        for i in remaining:
            candidate = [
                sp + [examples[i].get(ann_key, {})]
                for (ann_key, _, _), sp in zip(sources, selected_per_source)
            ]
            combined, _ = weighted_score(candidate)
            marginal     = combined - ensemble_score
            if marginal > best_marginal:
                best_marginal = marginal
                best_idx      = i

        for (ann_key, _, _), sp in zip(sources, selected_per_source):
            sp.append(examples[best_idx].get(ann_key, {}))

        selected_indices.append(best_idx)
        remaining.discard(best_idx)

        ensemble_score, covered = weighted_score(selected_per_source)
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
              f"ensemble%={ensemble_score/max_score*100:5.2f}% | "
              f"patterns={len(covered)}")

    return selected_indices, selection_log