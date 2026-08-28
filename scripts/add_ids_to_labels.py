"""
Add a unique `id` attribute to every <auto_label> and <manual_label> tag in each
HTML document, where id = its position (0, 1, 2, ...) among all such tags in
that document, in document order (nested tags included, parent before child).

Modifies files in place across:
    DATA_DIR / "annotated" / {dev, incoming, test, train}

If a tag already has an id attribute, it is overwritten.
"""

import chardet
from bs4 import BeautifulSoup

from configs.config import DATA_DIR

ANNOTATED_DIR = DATA_DIR / "annotated"
SUBFOLDERS = ["dev", "incoming", "test", "train"]
LABEL_TAGS = ["auto_label", "manual_label"]

ANNOTATED_DIR = DATA_DIR / "supplementary"
SUBFOLDERS = ["llm_extraction", "manual_extraction", "manual_verification"]


def add_ids_to_labels(html_file):
    """
    Detect encoding, read content, parse with BeautifulSoup, assign a
    sequential `id` to every auto_label/manual_label tag in document order,
    and return the modified HTML as a string (always re-serialized as UTF-8
    content; caller is responsible for writing with encoding='utf-8').
    """
    # Detect the encoding of the file
    with open(html_file, 'rb') as f:
        raw_data = f.read()
        detected = chardet.detect(raw_data)
        detected_encoding = detected['encoding']
        confidence = detected['confidence']
        print(f"  → Detected encoding: {detected_encoding} (confidence: {confidence:.2%})")

    # Read the original HTML content with detected encoding
    with open(html_file, 'r', encoding=detected_encoding) as f:
        content = f.read()

    # Parse the HTML
    soup = BeautifulSoup(content, 'html.parser')

    # Find every auto_label / manual_label tag, in document order
    label_tags = soup.find_all(LABEL_TAGS)

    if not label_tags:
        print("  → No auto_label/manual_label tags found, leaving file unchanged.")
        return None

    for position, tag in enumerate(label_tags):
        tag['id'] = str(position)

    print(f"  → Assigned ids 0..{len(label_tags) - 1} to {len(label_tags)} label tag(s).")

    return str(soup)


if __name__ == "__main__":

    for subfolder in SUBFOLDERS:
        folder_dir = ANNOTATED_DIR / subfolder

        if not folder_dir.exists():
            print(f"Skipping missing folder: {folder_dir}\n")
            continue

        html_files = list(folder_dir.glob("*.html")) + list(folder_dir.glob("*.htm"))
        print(f"[{subfolder}] Found {len(html_files)} HTML/HTM files to process.\n")

        for html_file in html_files:
            print(f"Processing: {html_file.name}")

            modified_html = add_ids_to_labels(html_file)

            if modified_html is None:
                print()
                continue

            # Overwrite the original file in place, as UTF-8
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(modified_html)

            print(f"✓ Updated in place: {html_file}\n")

    print("\n✓ Processing complete!")