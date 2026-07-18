"""
Plot coverage curves (surface or structural) for already-precomputed
greedy vs random few-shot selections.

Does NOT run any selection — only loads the two cached selection files
and replays coverage accumulation in the order examples already appear
in each file. Raises if either cache is missing.

Usage:
    python plot_fewshot_comparison.py --metric surface
    python plot_fewshot_comparison.py --metric structural
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from configs.config import FEWSHOT_CACHE_DIR, IMG_DIR, GREEDY_CONFIG
from src.fewshot.patterns.builder import (
    load_surface_pattern_dict,
    load_structural_pattern_dict,
    surface_pattern_dict_exists,
    structural_pattern_dict_exists,
)
# Adjust this import to wherever greedy_select_examples actually lives
from src.fewshot.selector.greedy import _KEY_MAP, _build_score_lookup



RANDOM_PATH = FEWSHOT_CACHE_DIR / "examples_random.json"

METRIC_TO_ANN_KEY = {
    "surface": "surface_pattern",
    "structural": "structural_pattern",
}


def _load_selection(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected cached selection at {path}, but it does not exist.\n"
            f"Run precompute_fewshot_examples.py first."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["examples"]


def _load_pattern_dict(metric: str) -> dict:
    if metric == "surface":
        if not surface_pattern_dict_exists():
            raise RuntimeError("Surface pattern dict not found. Run precompute_pattern_dict.py first.")
        return load_surface_pattern_dict()
    else:
        if not structural_pattern_dict_exists():
            raise RuntimeError("Structural pattern dict not found. Run precompute_pattern_dict.py first.")
        return load_structural_pattern_dict()


def _make_hashable(p):
    return tuple(p) if isinstance(p, list) else p


def _cumulative_coverage_curve(examples: list[dict], ann_key: str, score_lookup: dict) -> list[float]:
    """Cumulative (ever-growing) coverage score, replaying examples in file order."""
    seen = set()
    total = 0.0
    curve = []
    for ex in examples:
        label_patterns = ex.get(ann_key, {})
        for label_key, patterns in label_patterns.items():
            dict_key = _KEY_MAP.get(label_key)
            if dict_key is None:
                continue
            lookup = score_lookup[dict_key]
            for p in patterns:
                p = _make_hashable(p)
                key = (dict_key, p)
                if key not in seen:
                    seen.add(key)
                    total += lookup.get(p, 0.0)
        curve.append(total)
    return curve


def main(metric: str) -> None:
    if metric not in METRIC_TO_ANN_KEY:
        raise ValueError(f"metric must be one of {list(METRIC_TO_ANN_KEY)}")
    ann_key = METRIC_TO_ANN_KEY[metric]

    if metric == "surface":
        greedy_path = FEWSHOT_CACHE_DIR / f"examples_greedy_surf-{1.0}_struct-{0.0}.json"
    if metric == "structural":
        greedy_path = FEWSHOT_CACHE_DIR / f"examples_greedy_surf-{0.0}_struct-{1.0}.json"


    greedy_examples = _load_selection(greedy_path)
    random_examples = _load_selection(RANDOM_PATH)

    if len(greedy_examples) != len(random_examples):
        raise ValueError(
            f"Selections have different sizes: greedy={len(greedy_examples)}, "
            f"random={len(random_examples)}. Expected same size."
        )

    pattern_dict = _load_pattern_dict(metric)
    score_lookup = _build_score_lookup(pattern_dict)
    max_score = len(pattern_dict.keys()) * 100

    greedy_curve = _cumulative_coverage_curve(greedy_examples, ann_key, score_lookup)
    random_curve = _cumulative_coverage_curve(random_examples, ann_key, score_lookup)

    steps = list(range(1, len(greedy_curve) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(steps, [s / max_score * 100 for s in greedy_curve], label="greedy", marker="o", markersize=3)
    plt.plot(steps, [s / max_score * 100 for s in random_curve], label="random", marker="o", markersize=3)
    plt.xlabel("Number of examples selected")
    plt.ylabel(f"{metric.capitalize()} pattern coverage (%)")
    plt.title(f"{metric.capitalize()} coverage: greedy vs random")
    plt.legend()
    plt.grid(alpha=0.3)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = IMG_DIR / f"coverage_comparison_{metric}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✅ Plot saved → {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["surface", "structural"], default="surface")
    args = parser.parse_args()
    main(args.metric)