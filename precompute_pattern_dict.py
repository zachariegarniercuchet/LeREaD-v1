"""
Build and cache the pattern dict from train-set annotations.

Usage:
    python precompute_pattern_dict.py
    python precompute_pattern_dict.py --force
"""
import argparse
from config import DATA_DIR
from src.fewshot.patterns.builder import (
    build_pattern_dict, save_pattern_dict,
    load_pattern_dict, pattern_dict_exists,
)
from src.fewshot.patterns.ann_extractor import extract_parent_level_annotations


def _load_train_annotations() -> dict:
    """Merge annotations across all train HTML files."""
    merged = {"decision": [], "legislation": [], "secondary sources": []}
    folder = DATA_DIR / "annotated" / "train"
    for path in folder.glob("*.html"):
        try:
            html = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            html = path.read_text(encoding="latin-1")
        anns = extract_parent_level_annotations(html)
        for k in merged:
            merged[k].extend(anns.get(k, []))
    return merged


def main(force: bool) -> None:
    if not force and pattern_dict_exists():
        print("✓ Pattern dict already cached. Use --force to rebuild.")
        return
    print("🔄 Loading train annotations…")
    all_annotations = _load_train_annotations()
    print("🔄 Building pattern dict…")
    pattern_dict = build_pattern_dict(all_annotations)
    save_pattern_dict(pattern_dict)
    n_patterns = sum(len(v) for v in pattern_dict.values())
    print(f"✅ Saved — {n_patterns} patterns across {len(pattern_dict)} keys.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.force)