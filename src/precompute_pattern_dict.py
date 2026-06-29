"""
Build and cache the pattern dict from train-set annotations.

Usage:
    python precompute_pattern_dict.py
    python precompute_pattern_dict.py --force
"""
import argparse
from configs.config import DATA_DIR
from src.fewshot.patterns.builder import (
    build_surface_pattern_dict, save_surface_pattern_dict,
    load_surface_pattern_dict, surface_pattern_dict_exists,
    build_structural_pattern_dict, save_structural_pattern_dict,
    load_structural_pattern_dict, structural_pattern_dict_exists
)
from src.ann_extractor import extract_parent_level_annotations


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
        for rm in anns:
            labelname = rm.name
            if labelname in merged:
                merged[labelname].append({
                    "full_html": str(rm),
                    "docid": rm.html_tag.attributes["docid"],
                    "uri": rm.html_tag.attributes["uri"],
                    "text_content": rm.text,
                    "all_sublabels": rm.sublabels,
                })
    return merged


def main(force: bool, group: bool) -> None:

    surface_cached = force or not surface_pattern_dict_exists()
    structural_cached = force or not structural_pattern_dict_exists()

    if not surface_cached and not structural_cached:
        print("✓ Both pattern dicts already cached. Use --force to rebuild.")
        return

    print("🔄 Loading train annotations…")
    all_annotations = _load_train_annotations()

    if surface_cached:
        print("🔄 Building surface pattern dict…")
        surface_pattern_dict = build_surface_pattern_dict(all_annotations)
        save_surface_pattern_dict(surface_pattern_dict)
        n_patterns = sum(len(v) for v in surface_pattern_dict.values())
        print(f"✅ Saved — {n_patterns} patterns across {len(surface_pattern_dict)} keys.")

    if structural_cached:
        print("🔄 Building structural pattern dict…")
        structural_pattern_dict = build_structural_pattern_dict(all_annotations, group=group)
        save_structural_pattern_dict(structural_pattern_dict)
        n_patterns = sum(len(v) for v in structural_pattern_dict.values())
        print(f"✅ Saved — {n_patterns} patterns across {len(structural_pattern_dict)} keys.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--group", action="store_true", help="Group patterns by deduplicating consecutive sublabels")
    args = parser.parse_args()
    main(args.force, args.group)