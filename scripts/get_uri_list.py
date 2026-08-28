"""
extract_unique_uris.py
----------------------
Reads uri_dataset.json (produced by extract_uris.py) and outputs:
  - A summary of unique URI counts (printed to console)
  - uri_list.json : a sorted list of unique URIs only

Usage
-----
    python extract_unique_uris.py

Both files must be in the same Annotated folder.
"""

import json
from pathlib import Path
from bs4 import BeautifulSoup
from configs.config import SPLITS, DATA_DIR

# ── Configuration ─────────────────────────────────────────────────────────────

ANNOTATED_DIR = DATA_DIR / "annotated"

TARGET_LABELNAMES = {"legislation", "decision", "secondary sources"}
TAG_NAMES = ["manual_label", "auto_label"]
OUTPUT_FILE = ANNOTATED_DIR / "uris.json"

# ── Main ──────────────────────────────────────────────────────────────────────

def extract_from_file(filepath: Path) -> list[dict]:
    """Return a list of record dicts for every matching label tag in *filepath*."""
    records = []
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        soup = BeautifulSoup(fh, "html.parser")

    for tag_name in TAG_NAMES:
        for tag in soup.find_all(tag_name):
            labelname = tag.get("labelname", "")
            if labelname not in TARGET_LABELNAMES:
                continue

            records.append(
                {
                    "source_file": filepath.name,
                    "labelname":   labelname,
                    "docid":       tag.get("docid", None),
                    "uri":         tag.get("uri", None),
                    "text":        tag.get_text(strip=True),
                }
            )

    return records


def main():

    html_files = sorted(
        file
        for split in SPLITS
        for file in (ANNOTATED_DIR / split).glob("*.html")
        if (ANNOTATED_DIR / split).is_dir()
    )

    all_records: list[dict] = []

    for html_file in html_files:
        file_records = extract_from_file(html_file)
        all_records.extend(file_records)
        print(f"  {html_file.name:50s}  →  {len(file_records):4d} records")

    all_uris   = [r["uri"] for r in all_records]
    total      = len(all_uris)

    # Separate None / "None" from real URIs
    none_values = {"None", "none", "", None}
    real_uris   = [u for u in all_uris if u not in none_values]
    unique_uris = sorted(set(real_uris))

    none_count   = total - len(real_uris)
    unique_count = len(unique_uris)

    # Console summary
    print(f"Total records          : {total}")
    print(f"Records with no URI    : {none_count}")
    print(f"Records with a URI     : {len(real_uris)}")
    print(f"Unique URIs            : {unique_count}")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(unique_uris, out, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved {unique_count} unique URIs to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()