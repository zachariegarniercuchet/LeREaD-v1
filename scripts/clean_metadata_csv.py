import argparse
import csv
import json
import re
from pathlib import Path
from urllib.parse import urlparse


def extract_docs(metadata_text):
    """
    Parse the Solr JSON stored in the CSV metadata field
    and return response.docs.
    """
    try:
        data = json.loads(metadata_text)
    except json.JSONDecodeError as e:
        print("JSON parsing failed:")
        print(f"  {e}")
        print()
        print("First 1000 characters:")
        print(metadata_text[:1000])
        print()
        return None

    try:
        docs = data["response"]["docs"]
    except (KeyError, TypeError):
        print("Could not find response.docs")
        return None

    if not isinstance(docs, list):
        print("response.docs is not a list")
        return None

    return docs


def normalize_path(url):
    """
    Convert a full URL into its path.

    Example:
        https://www.canlii.org/en/ca/scc/doc/1992/1992canlii110/1992canlii110.html

    becomes:
        /en/ca/scc/doc/1992/1992canlii110/1992canlii110.html
    """
    if not url:
        return None

    parsed = urlparse(url)

    if parsed.path:
        return parsed.path.rstrip("/")

    return url


def process_file(input_file, output_file):
    input_file = Path(input_file)
    output_file = Path(output_file)

    output_rows = []

    detected = 0
    failed = 0

    with input_file.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        # Validate the expected input columns.
        expected_columns = {
            "original_url",
            "final_url",
            "http_status",
            "normalized_final_url",
            "solr_collection",
            "metadata",
        }

        if reader.fieldnames is None:
            raise ValueError("Could not find CSV header.")

        missing = expected_columns - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"Missing expected columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            detected += 1

            original_url = row["original_url"]
            normalized_final_url = row["normalized_final_url"]
            metadata_text = row["metadata"]

            docs = extract_docs(metadata_text)

            if docs is None:
                failed += 1
                print(
                    f"[WARNING] Could not parse metadata "
                    f"on CSV row {row_number}: "
                    f"{original_url}"
                )
                continue

            # Use normalized_final_url as the canonical path.
            normalized_path = normalize_path(
                normalized_final_url
            )

            for doc in docs:
                if normalized_path:
                    doc["path"] = normalized_path

            # The downstream pipeline expects metadata
            # to be a JSON array.
            metadata_json = json.dumps(
                docs,
                ensure_ascii=False,
                indent=2
            )

            output_rows.append(
                [
                    original_url,
                    metadata_json,
                ]
            )

            if detected % 100 == 0:
                print(
                    f"Processed {detected} records "
                    f"({len(output_rows)} successful)"
                )

    # Write the cleaned CSV.
    with output_file.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "original_url",
            "metadata",
        ])

        writer.writerows(output_rows)

    print()
    print("Finished.")
    print(f"Detected:             {detected}")
    print(f"Successfully cleaned: {len(output_rows)}")
    print(f"Failed:               {failed}")
    print(f"Output:               {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean CanLII Solr metadata CSV"
    )

    parser.add_argument(
        "input",
        help="Input CSV containing Solr metadata"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="cleaned.csv",
        help="Output cleaned CSV"
    )

    args = parser.parse_args()

    process_file(
        args.input,
        args.output
    )
