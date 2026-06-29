"""
Precompute the RPL (Reference Profile List) for all documents in the dataset and save it to a file.
"""
import json
from pathlib import Path

from configs.config import DATA_DIR, DATA_DIR, PROFILE_CACHE_DIR, SPLITS
from src.rpr import ReferenceProfileRegistry
from src.ann_extractor import extract_parent_level_annotations
from src.tokenizer_utils import decode, tokenize
from src.transforme_utils import clean_tokens
from src.rpr import ReferenceProfileRegistry


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

def _cache_path(cache_dir: str, split: str, filename: str) -> Path:
    return Path(cache_dir) / split / f"{filename}.json"


def cache_exists(cache_dir: str, split: str, filename: str) -> bool:
    return _cache_path(cache_dir, split, filename).is_file()


def build_registry_for_document(html: str) -> ReferenceProfileRegistry:
    """
    Walk every parent-level <manual_label>/<auto_label> mention in a
    document, in order, building up a ReferenceProfileRegistry as we go.

    Each ReferenceProfile in the resulting registry carries a
    `first_seen_id`: the id of the tag whose mention first caused that
    profile to be created. In other words, for any profile, every mention
    appearing before `first_seen_id` in the document did NOT yet have that
    profile in the registry.
    """
    mentions = extract_parent_level_annotations(html)

    registry = ReferenceProfileRegistry()
    for mention in mentions:
        registry.update_from_mention(mention)

    return registry


def _save_registry(cache_dir: str, split: str, filename: str, registry: ReferenceProfileRegistry) -> Path:
    path = _cache_path(cache_dir, split, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(registry.to_json(ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def precompute(splits: list[str], force: bool) -> None:
    # Collect pending files before doing any heavier work
    pending: list[tuple[str, str, str]] = []
    for split in splits:
        for filename, html in get_html_files(split).items():
            if not force and cache_exists(cache_dir=PROFILE_CACHE_DIR, split=split, filename=filename):
                print(f"  ✓ [{split}] {filename} — skip")
                continue
            pending.append((split, filename, html))

    if not pending:
        print("✅ All files already cached.")
        return

    for split, filename, html in pending:
        print(f"  → [{split}] {filename} … ", end="", flush=True)
        registry = build_registry_for_document(decode(clean_tokens(tokenize(html), keep_manual_label=True, keep_auto_label=True)))
        out_path = _save_registry(PROFILE_CACHE_DIR, split, filename, registry)

        print(f"✓ {len(registry)} profile(s) → {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    precompute(splits=args.splits, force=args.force)