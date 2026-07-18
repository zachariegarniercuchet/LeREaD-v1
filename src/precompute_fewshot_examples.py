"""
Select and cache few-shot examples.
Requires chunk cache and pattern dict to already exist (for the extraction task).

Usage:
    python precompute_fewshot_examples.py
    python precompute_fewshot_examples.py --task extraction --method random --n 20
    python precompute_fewshot_examples.py --task extraction --force
    python precompute_fewshot_examples.py --task coref --n 20
    python precompute_fewshot_examples.py --task coref --force
"""
import argparse
import json
from configs.config import DATA_DIR, FEWSHOT_CACHE_DIR, FEWSHOT_CONTEXT_MAX_TOKENS, FEWSHOT_N, FEWSHOT_METHOD, FEWSHOT_MAX_INPUT_LEN, FS_MIN_TOKENS, IMG_DIR, PROFILE_CACHE_DIR
from src.ann_extractor import extract_parent_level_annotations
from src.fewshot.patterns.builder import load_surface_pattern_dict, load_structural_pattern_dict, structural_pattern_dict_exists, surface_pattern_dict_exists
from src.fewshot import greedy_select_examples, random_select_examples
from src.plotting_utils import plot_coverage_comparison
from src.transforme_utils import LabelTransformConfig, clean_tokens
from configs.config import GREEDY_CONFIG
from pathlib import Path
from src.extractor import decode, tokenize
from src.ann_extractor import get_mention_upper_context
from src.transforme_utils import prepare_label_tokens
from src.rpr import ReferenceProfileRegistry
from tqdm import tqdm
import copy


TASKS = ["extraction", "coref"]


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


def _load_extraction_examples(fs_min_tokens: int) -> list[dict]:
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


def _load_coref_examples() -> list[dict]:
    """Chunk train HTML files and extract few-shot (input, output) pairs
    for the coreference resolution task.
    """

    input_label_config = LabelTransformConfig(
        use_simplified=True,
        switch_type=False,
        keep_attributes=["labelname"],
    )

    output_label_config = LabelTransformConfig(
        use_simplified=True,
        switch_type=False,
        keep_attributes=["labelname", "docid"],
    )

    examples = []
    for filename, html in get_html_files("train").items():


        cache_path = PROFILE_CACHE_DIR / "train" / f"{Path(filename).stem}.json"
        if not cache_path.exists():
            continue
        with open(cache_path, "r", encoding="utf-8") as f:
            registry_dict = json.load(f)

        rpr = ReferenceProfileRegistry.from_dict(registry_dict)


        mentions = extract_parent_level_annotations(decode(clean_tokens(tokenize(html), keep_manual_label=True, keep_auto_label=True)))

        for mention in tqdm(mentions, desc=f"Processing mentions {filename}", unit="mention"):
            mention_id_str = mention.html_tag.attributes.get("id")
            if mention_id_str is None:
                continue
            mention_id = int(mention_id_str)

            docid = mention.html_tag.attributes["docid"]

            profile = rpr.get_profile_by_docid(docid)
            if profile is None:
                print(f"Warning: docid {docid} not found in registry for mention id {mention_id}. Skipping this mention.")
                continue


            context = get_mention_upper_context(decode(clean_tokens(tokenize(html), keep_manual_label=False, keep_auto_label=False, protected_id=mention_id)), mention, max_tokens=FEWSHOT_CONTEXT_MAX_TOKENS)

            input_mention = decode(prepare_label_tokens(tokenize(mention.html_str), input_label_config))

            #mention.html_tag.set_attribute("docid", profile.main_title)  # replace docid by the main_title

            output_mention = decode(prepare_label_tokens(tokenize(mention.html_str), output_label_config))

            snapshot_filtered = rpr._filter_registry_before(mention_id)
            snapshot_filtered = snapshot_filtered._filter_by_doctype(mention.html_tag.name)  # Filter the snapshot by the mention's doc_type

            examples.append({
                "input": json.dumps({"input_mention": input_mention, "profileRegistry": snapshot_filtered.to_dict(), "context": context}, ensure_ascii=False),
                "output": json.dumps(output_mention, ensure_ascii=False),
                "meta": {"docid": docid, "mention_id": mention_id, "has_main_title": 'titletype="official"' in mention.html_str}
            })

        #break

    return examples


def _filter(examples, max_len):
    return [ex for ex in examples if len(ex["example"]["input"]) < max_len]





def _run_extraction(method: str, n: int, force: bool) -> None:


    example_path = FEWSHOT_CACHE_DIR / f"examples_{method}_surf-{GREEDY_CONFIG['surface_pattern']}_struct-{GREEDY_CONFIG['structural_pattern']}.json"

    if not force and example_path.is_file():
        print("✓ Few-shot examples already cached. Use --force to rebuild.")
        return

    if method == "greedy" and not surface_pattern_dict_exists():
        raise RuntimeError("Surface pattern dict not found. Run precompute_pattern_dict.py first.")

    examples = _filter(_load_extraction_examples(fs_min_tokens=FS_MIN_TOKENS), FEWSHOT_MAX_INPUT_LEN)
    print(f"Candidate pool: {len(examples)} examples after filtering.")

    if method == "greedy":
        surface_pattern_dict = load_surface_pattern_dict()
        structural_pattern_dict = load_structural_pattern_dict()
        pattern_sources = [("surface_pattern", surface_pattern_dict, GREEDY_CONFIG["surface_pattern"]), ("structural_pattern", structural_pattern_dict, GREEDY_CONFIG["structural_pattern"])]
        indices, log = greedy_select_examples(examples, pattern_sources=pattern_sources, n=n)
    else:
        indices, log = random_select_examples(examples, n=n)

    selected = [examples[i] for i in indices]
    example_path.parent.mkdir(parents=True, exist_ok=True)
    with example_path.open("w", encoding="utf-8") as f:
        json.dump({"method": method, "n": n, "surface_weight": GREEDY_CONFIG["surface_pattern"], "structural_weight": GREEDY_CONFIG["structural_pattern"], "examples": selected, "log": log}, f, indent=2)
    print(f"✅ Saved {len(selected)} examples → {example_path}")


def _run_coref(n: int, force: bool) -> None:
    # Coref only supports random selection.
    example_path = FEWSHOT_CACHE_DIR / "examples_coref_random.json"

    if not force and example_path.is_file():
        print("✓ Few-shot coref examples already cached. Use --force to rebuild.")
        return

    examples = _load_coref_examples()
    print(f"Candidate pool: {len(examples)} examples.")

    indices, log = random_select_examples(examples, n=n)
    selected = [examples[i] for i in indices]

    example_path.parent.mkdir(parents=True, exist_ok=True)
    with example_path.open("w", encoding="utf-8") as f:
        json.dump({"task": "coref", "method": "random", "n": n, "examples": selected, "log": log}, f, indent=2)
    print(f"✅ Saved {len(selected)} examples → {example_path}")


def main(task: str, method: str, n: int, force: bool) -> None:
    if task == "coref":
        if method == "greedy":
            print("⚠  Coref task only supports random selection. Ignoring --method greedy and using random.")
        _run_coref(n, force)
        return

    # task == "extraction"
    _run_extraction(method, n, force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="extraction", choices=TASKS, help="Which few-shot task to precompute examples for.")
    parser.add_argument("--method", default=FEWSHOT_METHOD, choices=["greedy", "random"], help="Selection method. Forced to 'random' for --task coref.")
    parser.add_argument("--n", type=int, default=FEWSHOT_N)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.task, args.method, args.n, args.force)