"""
Extract and collect sublabel strings from annotated HTML files.
"""
from bs4 import BeautifulSoup
from src.htmlLabel import ReferenceMention
from src.tokenizer_utils import decode, tokenize


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

    return [rm for rm in (ReferenceMention(str(parent)) for parent in parent_labels)]


def get_mention_upper_context(html: str, mention, max_tokens: int = 500) -> str:
    """
    Find `mention` inside `html`, and return up to `max_tokens` tokens
    of context immediately preceding it, decoded back to a string.

    Args:
        html: the full document html.
        mention: a mention object with `.html_tag` (has `.attributes["id"]`)
                  and `.html_str` (the raw html snippet for this mention).
        max_tokens: maximum number of preceding tokens to include.

    Returns:
        Decoded string of the context window preceding the mention.
    """
    tokens = tokenize(html)

    start_idx = _find_mention_token_start(tokens, mention)

    if start_idx is None:
        raise ValueError(f"Could not locate mention with id={mention.html_tag.attributes.get('id')} in tokenized html")

    context_start = max(0, start_idx - max_tokens)
    context_tokens = tokens[context_start:start_idx]

    return decode(context_tokens)


def _find_mention_token_start(tokens, mention):
    """
    Locate the index of the first token belonging to `mention` inside `tokens`.
    """
    tag = mention.html_tag

    for i, tok in enumerate(tokens):


        if str(tag) == tok:
            return i

    return None
    


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