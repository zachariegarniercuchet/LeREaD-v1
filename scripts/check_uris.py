#!/usr/bin/env python3
"""
Check that every non-None `uri` found on <manual_label>/<auto_label> tags in the
annotated HTML documents exists in the candidate metadata CSV (column `original_url`).

Usage:
    python check_uris.py
    python check_uris.py --metadata metadata.csv --root data/annotated \
                         --splits train dev test incoming --report missing_uris.csv
"""
import argparse
import csv
import sys
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path

LABEL_TAGS = {"manual_label", "auto_label"}
EMPTY_URIS = {"", "none", "null", "nan"}


def normalize(uri: str) -> str:
    """Light normalisation so trivial differences don't cause false alarms."""
    u = uri.strip().lower()
    if u.startswith("http://"):
        u = "https://" + u[len("http://"):]
    u = u.replace("://www.", "://")
    return u.rstrip("/")


class LabelParser(HTMLParser):
    """Collects the `uri` attribute of every manual_label / auto_label tag."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.uris = []  # list of (uri, labelname, docid)

    def handle_starttag(self, tag, attrs):
        if tag.lower() not in LABEL_TAGS:
            return
        a = dict(attrs)
        uri = a.get("uri")
        if uri is None or uri.strip().lower() in EMPTY_URIS:
            return
        self.uris.append((uri.strip(), a.get("labelname", ""), a.get("docid", "")))


def load_metadata_uris(path: Path) -> set:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # metadata column can be large
    uris = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = (row.get("original_url") or "").strip()
            if url:
                uris.add(normalize(url))
    return uris


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="metadata.csv")
    ap.add_argument("--root", default="data/annotated")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test", "incoming"])
    ap.add_argument("--report", default="missing_uris.csv",
                    help="CSV listing every missing uri with its file")
    args = ap.parse_args()

    known = load_metadata_uris(Path(args.metadata))
    print(f"Loaded {len(known)} candidate URIs from {args.metadata}\n")

    report_rows = []
    all_ok = True

    for split in args.splits:
        split_dir = Path(args.root) / split
        if not split_dir.is_dir():
            print(f"[{split}] directory not found: {split_dir}\n")
            all_ok = False
            continue

        files = sorted(p for p in split_dir.rglob("*") if p.suffix.lower() in {".html", ".htm"})
        n_labels_with_uri = 0
        distinct = set()
        missing = defaultdict(list)  # normalized uri -> [(file, raw uri, labelname, docid)]

        for fp in files:
            parser = LabelParser()
            parser.feed(fp.read_text(encoding="utf-8", errors="replace"))
            parser.close()
            for raw, labelname, docid in parser.uris:
                n_labels_with_uri += 1
                norm = normalize(raw)
                distinct.add(norm)
                if norm not in known:
                    missing[norm].append((fp, raw, labelname, docid))

        ok = not missing
        all_ok &= ok
        print(f"[{split}] {'OK ✅' if ok else 'MISSING ❌'}")
        print(f"  files: {len(files)} | labels with a uri: {n_labels_with_uri} | "
              f"distinct uris: {len(distinct)} | missing distinct uris: {len(missing)}")
        for norm, occ in sorted(missing.items()):
            print(f"    - {occ[0][1]}  ({len(occ)} occurrence(s), e.g. {occ[0][0].name})")
            for fp, raw, labelname, docid in occ:
                report_rows.append([split, str(fp), raw, labelname, docid])
        print()

    if report_rows:
        with open(args.report, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split", "file", "uri", "labelname", "docid"])
            w.writerows(report_rows)
        print(f"Detailed report written to {args.report}")

    print("ALL URIs FOUND IN METADATA ✅" if all_ok else "Some URIs are missing from metadata ❌")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()