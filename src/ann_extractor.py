"""
Extract and collect sublabel strings from annotated HTML files.
"""
from bs4 import BeautifulSoup




def extract_parent_level_annotations(html_content: str) -> dict:
    """
    Returns {'decision': [...], 'legislation': [...], 'secondary sources': [...]}
    Each entry is a list of annotation dicts with keys:
      full_html, docid, uri, text_content, direct_sublabels, all_sublabels.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    parent_labels = soup.find_all(
        ["manual_label", "auto_label"], attrs={"parent": ""}
    )
    annotations = {"decision": [], "legislation": [], "secondary sources": []}

    idx = 0
    for label in parent_labels:
        labelname = label.get("labelname", "")
        if labelname not in annotations:
            continue
        direct = [sl.get("labelname", "")
                  for sl in label.find_all(["manual_label", "auto_label"],
                                           recursive=False)]
        all_sl = [sl.get("labelname", "")
                  for sl in label.find_all(["manual_label", "auto_label"])]
        annotations[labelname].append({
            "full_html":       str(label),
            "docid":           label.get("docid", ""),
            "uri":             label.get("uri", ""),
            "text_content":    label.get_text(strip=True),
            "direct_sublabels": direct,
            "all_sublabels":   all_sl,
            "order": idx,
        })
        idx += 1
    return annotations


def get_sublabel_strings(
    annotations_dict: dict,
    parent_label: str,
    sublabel_key: str,
    max_items: int | None = None,
) -> list[str]:
    """
    Collect raw text strings for one sublabel type across all annotations.
    Works with both manual_label HTML and JSON-style XML output.
    """
    results = []
    for ann in annotations_dict.get(parent_label, []):
        html = ann.get("full_html", "")
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        if "manual_label" in html or "auto_label" in html:
            elements = soup.find_all(
                ["manual_label", "auto_label"], attrs={"labelname": sublabel_key}
            )
        else:
            elements = soup.find_all(sublabel_key)
        for el in elements:
            text = el.get_text(strip=True)
            if text:
                results.append(text)
                if max_items and len(results) >= max_items:
                    return results
    return results