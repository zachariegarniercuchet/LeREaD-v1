"""
Select and cache few-shot examples.
Requires chunk cache and pattern dict to already exist.

Usage:
    python precompute_fewshot_examples.py
    python precompute_fewshot_examples.py --method random --n 20
    python precompute_fewshot_examples.py --force
    python precompute_fewshot_examples.py --compare  # Generate both greedy & random + comparison plot
"""
import argparse
import json
from configs.config import DATA_DIR, FEWSHOT_CACHE_DIR, FEWSHOT_N, FEWSHOT_METHOD, FEWSHOT_MAX_INPUT_LEN, FS_MIN_TOKENS, IMG_DIR, KEEP_ATRIBUTES, USE_SIMPLIFIED_LABELS
from src.fewshot.patterns.builder import load_surface_pattern_dict, load_structural_pattern_dict, surface_pattern_dict_exists, structural_pattern_dict_exists, save_surface_pattern_dict, save_structural_pattern_dict
from src.fewshot import greedy_select_examples, random_select_examples
from src.plotting_utils import plot_coverage_comparison
from src.transforme_utils import LabelTransformConfig
from configs.config import GREEDY_CONFIG


def get_html_files(split: str) -> dict[str, str]:
    folder = DATA_DIR / "annotated" / split
    if not folder.is_dir():
        print(f"⚠  Not found, skipping: {folder}")
        return {}
    files = {}
    for path in folder.iterdir():
        if path.suffix.lower() not in {".html", ".htm"}:
            continue
        files[path.stem] = path.read_text(encoding="utf-8")
    return files


def _load_candidate_examples(fs_min_tokens: int) -> list[dict]:
    """Chunk train HTML files and extract few-shot (input, output) pairs."""
    from src.chunkers.factory import ChunkerFactory
    from src.extractor import extract_few_shot_examples

    # Create input label config for processing
    input_label_config = LabelTransformConfig(
        use_simplified=False,
        switch_type=False,
        remove_attributes=["verified", "style"],
    )

    examples = []
    for filename, html in get_html_files("train").items():
        chunks = ChunkerFactory.get_chunks(
            html, method="paragraph",
            filename=filename, min_tokens=fs_min_tokens
        )
        examples.extend(extract_few_shot_examples(chunks, input_label_config, source_file=filename))
    return examples


def _filter(examples, max_len):
    return [ex for ex in examples if len(ex["example"]["input"]) < max_len]

def _generate_comparison(n: int) -> None:
    """Generate both greedy and random selections, then create comparison plot."""
    
    # Check if pattern dict exists for greedy
    if not surface_pattern_dict_exists() :
        raise RuntimeError("Pattern dict not found. Run precompute_pattern_dict.py first.")
    
    # Load candidate examples
    examples = _filter(_load_candidate_examples(fs_min_tokens=FS_MIN_TOKENS), FEWSHOT_MAX_INPUT_LEN)
    print(f"Candidate pool: {len(examples)} examples after filtering.")
    
    # Generate greedy selection
    print("\n--- Running greedy selection ---")
    pattern_dict = load_surface_pattern_dict()
    greedy_indices, greedy_log = greedy_select_examples(examples, pattern_dict, n=n)
    greedy_selected = [examples[i] for i in greedy_indices]
    
    # Generate random selection
    print("\n--- Running random selection ---")
    random_indices, random_log = random_select_examples(examples, n=n)
    random_selected = [examples[i] for i in random_indices]
    
    # Save both selections
    greedy_path = FEWSHOT_CACHE_DIR / f"examples_greedy.json"
    random_path = FEWSHOT_CACHE_DIR / f"examples_random.json"
    
    FEWSHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    with greedy_path.open("w", encoding="utf-8") as f:
        json.dump({"method": "greedy", "n": n, "examples": greedy_selected, "log": greedy_log}, f, indent=2)
    print(f"✅ Saved {len(greedy_selected)} greedy examples → {greedy_path}")
    
    with random_path.open("w", encoding="utf-8") as f:
        json.dump({"method": "random", "n": n, "examples": random_selected, "log": random_log}, f, indent=2)
    print(f"✅ Saved {len(random_selected)} random examples → {random_path}")
    
    # Generate comparison plot
    print("\n--- Generating comparison plot ---")
    plot_path = IMG_DIR / "coverage_comparison.png"
    plot_coverage_comparison(greedy_log, random_log, save_path=plot_path, 
                            examples=examples, pattern_dict=pattern_dict)
    

def main(method: str, n: int, force: bool, compare: bool = False) -> None:

    if compare:
        # Generate comparison plot for both greedy and random
        _generate_comparison(n)
        return

    example_path = FEWSHOT_CACHE_DIR / f"examples_{method}_surf-{GREEDY_CONFIG["surface_pattern"]}_struct-{GREEDY_CONFIG["structural_pattern"]}.json"

    if not force and example_path.is_file():
        print("✓ Few-shot examples already cached. Use --force to rebuild.")
        return

    if method == "greedy" and not surface_pattern_dict_exists():
        raise RuntimeError("Surface pattern dict not found. Run precompute_pattern_dict.py first.")

    examples = _filter(_load_candidate_examples(fs_min_tokens=FS_MIN_TOKENS), FEWSHOT_MAX_INPUT_LEN)
    print(f"Candidate pool: {len(examples)} examples after filtering.")

    if method == "greedy":
        surface_pattern_dict = load_surface_pattern_dict()
        structural_pattern_dict = load_structural_pattern_dict()
        pattern_sources = [( "surface_pattern", surface_pattern_dict, GREEDY_CONFIG["surface_pattern"] ), ( "structural_pattern", structural_pattern_dict, GREEDY_CONFIG["structural_pattern"] )]
        indices, log = greedy_select_examples(examples, pattern_sources=pattern_sources, n=n)
    else:
        indices, log = random_select_examples(examples, n=n)

    selected = [examples[i] for i in indices]
    example_path.parent.mkdir(parents=True, exist_ok=True)
    with example_path.open("w", encoding="utf-8") as f:
        json.dump({"method": method, "n": n, "surface_weight": GREEDY_CONFIG["surface_pattern"], "structural_weight": GREEDY_CONFIG["structural_pattern"], "examples": selected, "log": log}, f, indent=2)
    print(f"✅ Saved {len(selected)} examples → {example_path}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=FEWSHOT_METHOD, choices=["greedy", "random"])
    parser.add_argument("--n",      type=int, default=FEWSHOT_N)
    parser.add_argument("--force",  action="store_true")
    parser.add_argument("--compare", action="store_true", help="Generate both greedy and random selections with comparison plot")
    args = parser.parse_args()
    main(args.method, args.n, args.force, compare=args.compare)