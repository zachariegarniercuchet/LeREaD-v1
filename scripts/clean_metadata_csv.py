import csv
import json
import re
import argparse
from pathlib import Path


def clean_metadata_text(lines):
    parts = []
    for line in lines:
        line = line.rstrip("\r\n")
        line = re.sub(r";+$", "", line)
        if line.startswith('"'):
            line = line[1:]
        if line.endswith('"'):
            line = line[:-1]
        parts.append(line)

    text = "".join(parts)
    text = re.sub(r'"{2,}', '"', text)
    text = text.strip()

    # NEW: remove a leftover single quote that sits directly against the
    # JSON's structural braces. This is the real outer CSV-field quote
    # (present once at the very start/end of the whole blob) which the
    # per-line stripping above can't fully remove, because on the first/last
    # line it's stacked next to that line's own wrapper quote, and a lone
    # quote isn't caught by the "{2,}" collapse above.
    text = re.sub(r'^"+(?=\{)', '', text)
    text = re.sub(r'(?<=\})"+$', '', text)

    return text.strip()


def extract_docs(metadata_text):
    """
    Parse Solr JSON and extract response.docs.
    """

    try:
        data = json.loads(metadata_text)
    except json.JSONDecodeError as e:
        print("JSON parsing failed:")
        print(e)
        print()
        print("First 1000 characters:")
        print(metadata_text[:1000])
        print()
        return None

    try:
        return data["response"]["docs"]
    except (KeyError, TypeError):
        print("Could not find response.docs")
        return None


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

    match = re.match(r"https?://[^/]+(/.*)$", url)

    if match:
        return match.group(1).rstrip("/")

    return url


def parse_record_start(line):
    """
    Parse the first line of a record.

    The first line looks like:

    "https://canlii.ca/t/52lw2,https://www.canlii.org/...,200,
    https://www.canlii.org/...,canlii_frontend_en,""{
    """

    line = line.rstrip("\r\n")

    # Remove the first quote.
    if line.startswith('"'):
        line = line[1:]

    # We only split the first five commas.
    parts = line.split(",", 5)

    if len(parts) != 6:
        return None

    return {
        "original_url": parts[0],
        "final_url": parts[1],
        "http_status": parts[2],
        "normalized_final_url": parts[3],
        "solr_collection": parts[4],
        "metadata_start": parts[5],
    }


def is_record_start(line):
    """
    A new record always starts with a CanLII short URL.
    """

    return line.startswith('"https://canlii.ca/t/')


def process_file(input_file, output_file):

    input_file = Path(input_file)
    output_file = Path(output_file)

    with input_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    records = []
    current_record = None

    for line in lines:

        # Skip header.
        if line.startswith("original_url"):
            continue

        # New record.
        if is_record_start(line):

            if current_record is not None:
                records.append(current_record)

            parsed = parse_record_start(line)

            if parsed is None:
                print("WARNING: Could not parse record start:")
                print(line[:300])
                current_record = None
                continue

            current_record = {
                "original_url": parsed["original_url"],
                "final_url": parsed["final_url"],
                "normalized_final_url": parsed["normalized_final_url"],
                "metadata_lines": [
                    parsed["metadata_start"]
                ],
            }

        else:

            # Continuation of metadata.
            if current_record is not None:
                current_record["metadata_lines"].append(line)

    # Add last record.
    if current_record is not None:
        records.append(current_record)

    print(f"Detected {len(records)} records.")

    output_rows = []
    failed = 0

    for i, record in enumerate(records, start=1):

        metadata_text = clean_metadata_text(
            record["metadata_lines"]
        )

        docs = extract_docs(metadata_text)

        if docs is None:
            failed += 1
            print(
                f"[WARNING] Could not parse metadata "
                f"for record {i}: "
                f"{record['original_url']}"
            )
            continue

        # Use normalized_final_url as the canonical source
        # for the path.
        normalized_path = normalize_path(
            record["normalized_final_url"]
        )

        for doc in docs:

            if normalized_path:
                doc["path"] = normalized_path

        # The existing pipeline expects metadata to be
        # a JSON array.
        metadata_json = json.dumps(
            docs,
            ensure_ascii=False,
            indent=2
        )

        output_rows.append(
            [
                record["original_url"],
                metadata_json,
            ]
        )

        # Progress indicator.
        if i % 100 == 0:
            print(
                f"Processed {i}/{len(records)} "
                f"({len(output_rows)} successful)"
            )

    # Write the clean CSV.
    with output_file.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "original_url",
            "metadata"
        ])

        writer.writerows(output_rows)

    print()
    print("Finished.")
    print(f"Detected:             {len(records)}")
    print(f"Successfully cleaned: {len(output_rows)}")
    print(f"Failed:               {failed}")
    print(f"Output:               {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Clean corrupted CanLII metadata CSV"
    )

    parser.add_argument(
        "input",
        help="Input corrupted CSV"
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
