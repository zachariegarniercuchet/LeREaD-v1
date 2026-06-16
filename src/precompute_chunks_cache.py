"""
Precompute and cache chunks for all HTML files.

Usage:
    python precompute_chunks_cache.py
    python precompute_chunks_cache.py --splits train dev --method paragraph
    python precompute_chunks_cache.py --force
"""

import argparse
from configs.config import DATA_DIR, SPLITS
from src.chunkers import ChunkerFactory


def get_html_files(split: str) -> dict[str, str]:
    folder = DATA_DIR / "original" / split
    if not folder.is_dir():
        print(f"⚠  Not found, skipping: {folder}")
        return {}
    files = {}
    for path in folder.iterdir():
        if path.suffix.lower() not in {".html", ".htm"}:
            continue
        try:
            files[path.stem] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            files[path.stem] = path.read_text(encoding="latin-1")
    return files


def precompute(splits: list[str], method: str, force: bool) -> None:
    from src.chunkers.cache import cache_exists

    # Collect pending files before loading spaCy
    pending: list[tuple[str, str, str]] = []
    for split in splits:
        for filename, html in get_html_files(split).items():
            if not force and cache_exists(method, split, filename):
                print(f"  ✓ [{split}] {filename} — skip")
                continue
            pending.append((split, filename, html))

    if not pending:
        print("✅ All files already cached.")
        return

    # Load spaCy only if needed
    nlp = None
    if method == "sentence":
        import spacy
        print(f"\n🔄 Loading spaCy ({len(pending)} files to process)…")
        nlp = spacy.load("en_core_web_trf")
        print("✅ Model loaded.\n")

    for split, filename, html in pending:
        print(f"  → [{split}] {filename} … ", end="", flush=True)

        ChunkerFactory.get_chunks(
            html, method=method, split=split, filename=filename, nlp=nlp
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits",  nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--method",  default="sentence", choices=["sentence", "paragraph"])
    parser.add_argument("--force",   action="store_true")
    args = parser.parse_args()
    precompute(args.splits, args.method, args.force)