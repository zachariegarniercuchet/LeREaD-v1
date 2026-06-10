"""
src/fewshot/extractor.py

Builds (input, output) few-shot pairs from annotated HTML token chunks,
with metadata: structural annotation patterns and normalised sublabel patterns.

Public entry points:
  - extract_few_shot_examples             : chunk-level extraction (flat pass)
  - extract_few_shot_examples_from_labels : parent-label-scoped extraction

Each returned example dict has the shape:
    {
        "example":       {"input": str, "output": str},
        "source_file":   str,
        "comments":      "",
        "selected":      False,
        "notation":      0,
        "annotation_pattern":  [[parent, sub, sub, ...], ...],   # structural
        "label_pattern":       {                                  # normalised
            "decision_fragment":          [(...), ...],
            "decision_citation":          [(...), ...],
            "legislation_fragment":       [(...), ...],
            "legislation_citation":       [(...), ...],
            "secondary sources_fragment": [(...), ...],
            "secondary sources_source":   [(...), ...],
            "decision_title":             [(...), ...],
        },
    }
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from src import clean_tokens
from src import decode
from src import HTMLLabel
from src import is_manual_label_tag, is_auto_label_tag
from src.transforme_utils import prepare_label_tokens
from .fewshot.patterns.normalizers import (
    normalize_fragment,
    normalize_decision_citation,
    normalize_legislation_citation,
    normalize_secondary_source,
    normalize_decision_title,
)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LabelTransformConfig:
    """
    Controls HOW label tags are rendered in output tokens.

    Attributes:
        use_simplified:    Emit <title> instead of <manual_label labelname="title">.
        switch_type:       Swap manual_label ↔ auto_label on every tag.
        keep_attributes:   Attribute whitelist (all others dropped). Mutually
                           exclusive with remove_attributes.
        remove_attributes: Attribute blacklist. Mutually exclusive with
                           keep_attributes.
        keep_labels:       Label-name whitelist — tags not in this list are
                           stripped (content kept). Mutually exclusive with
                           remove_labels.
        remove_labels:     Label-name blacklist. Mutually exclusive with
                           keep_labels.
    """
    use_simplified:    bool             = False
    switch_type:       bool             = False
    keep_attributes:   list[str] | None = None
    remove_attributes: list[str] | None = None
    keep_labels:       list[str] | None = None
    remove_labels:     list[str] | None = None


@dataclass
class FewShotExtractionConfig:
    """
    Controls WHAT to extract when building few-shot pairs from parent labels.

    Attributes:
        parent_labels:   Labels that delimit each example span
                         (e.g. ["decision", "legislation"]).
        new_labels:      Sublabels shown *only* in the output (the labelling
                         task the model must learn).
        already_labeled: Sublabels shown in *both* input and output (already
                         annotated context the model may rely on).
        transform:       How all kept tags are rendered.
    """
    parent_labels:   list[str]
    new_labels:      list[str]
    already_labeled: list[str]            = field(default_factory=list)
    transform:       LabelTransformConfig = field(default_factory=LabelTransformConfig)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PARENT_LABEL_NAMES: frozenset[str] = frozenset(
    {"decision", "legislation", "secondary sources"} # to use tag.soup.find_all("secondary") instead of soup.find_all("secondary sources")
)

# (parent_label, child_label, normalizer_fn)
_SUBLABEL_PATHS: list[tuple[str, str, callable]] = [
    ("decision",          "fragment", normalize_fragment),
    ("legislation",       "fragment", normalize_fragment),
    ("secondary sources", "fragment", normalize_fragment),
    ("decision",          "citation", normalize_decision_citation),
    ("legislation",       "citation", normalize_legislation_citation),
    ("secondary sources", "source",   normalize_secondary_source),
    ("decision",          "title",    normalize_decision_title),
]





# ---------------------------------------------------------------------------
# Internal helper — spacing fix
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"<[^>]+>|\s+|[^<\s]+")


def _fix_tag_spacing(text: str) -> str:
    """
    Move leading whitespace that immediately follows an opening tag to
    *before* the tag, so that mentions never start with a space.

    Example:
        '<decision> word</decision>'  →  ' <decision>word</decision>'
    """
    tokens = _TOKEN_RE.findall(text)
    i = 0
    while i < len(tokens) - 1:
        cur, nxt = tokens[i], tokens[i + 1]
        if cur.startswith("<") and not cur.startswith("</") and nxt.isspace():
            tokens[i], tokens[i + 1] = nxt, cur
            i += 2
        else:
            i += 1
    return "".join(tokens)


# ---------------------------------------------------------------------------
# Internal helper — annotation pattern extraction
# ---------------------------------------------------------------------------

def _extract_annotation_pattern(output_text: str) -> list[list[str]]:
    """
    Return the structural annotation pattern of *output_text*.

    Works with both full format (``<manual_label labelname="decision">``) and
    simplified format (``<decision>``).

    Each entry in the returned list is a list whose first element is the
    parent label name and whose remaining elements are the sublabel names
    found inside it, in document order.

    Example return value::

        [["decision", "title", "citation"], ["legislation", "fragment"]]
    """
    try:
        return _extract_pattern_bs4(output_text)
    except Exception:
        return _extract_pattern_regex(output_text)


def _extract_pattern_bs4(output_text: str) -> list[list[str]]:
    soup = BeautifulSoup(output_text, "html.parser")
    patterns: list[list[str]] = []

    # --- full format: <manual_label labelname="decision"> ---
    for label in soup.find_all(["manual_label", "auto_label"]):
        name = label.get("labelname", "")
        if name not in _PARENT_LABEL_NAMES:
            continue
        sublabels = [
            child.get("labelname", "")
            for child in label.find_all(["manual_label", "auto_label"])
        ]
        patterns.append([name] + sublabels)

    if patterns:
        return patterns

    # --- simplified format: <decision>, <title>, … ---
    for parent_name in _PARENT_LABEL_NAMES:
        if " " in parent_name:
            # Use regex to find tags with spaces (e.g., <secondary source>...</secondary source>)
            escaped_name = re.escape(parent_name)
            tag_re = re.compile(rf"<{escaped_name}>(.*?)</{escaped_name}>", re.DOTALL | re.IGNORECASE)
            for m in tag_re.finditer(output_text):
                parent_tag = BeautifulSoup(m.group(0), "html.parser").find(True)
                if parent_tag:
                    sublabels = [
                        child.name
                        for child in parent_tag.find_all(True)
                        if child.name not in _PARENT_LABEL_NAMES
                    ]
                    patterns.append([parent_name] + sublabels)
        else:
            for parent_tag in soup.find_all(parent_name):
                # every child tag (any name) that is NOT itself a known parent
                sublabels = [
                    child.name
                    for child in parent_tag.find_all(True)          # recursive
                    if child.name not in _PARENT_LABEL_NAMES
                ]
                patterns.append([parent_name] + sublabels)

    return patterns


def _extract_pattern_regex(output_text: str) -> list[list[str]]:
    """Regex fallback — handles both full and simplified formats."""
    patterns: list[list[str]] = []
    parent_names_re = "|".join(re.escape(n) for n in _PARENT_LABEL_NAMES)

    # Full format
    full_parent_re = re.compile(
        rf'<(?:manual|auto)_label\s+labelname="({parent_names_re})">(.*?)</(?:manual|auto)_label>',
        re.DOTALL | re.IGNORECASE,
    )
    child_re = re.compile(r'<(?:manual|auto)_label\s+labelname="([^"]+)"', re.IGNORECASE)

    for m in full_parent_re.finditer(output_text):
        parent = m.group(1)
        sublabels = [c.group(1) for c in child_re.finditer(m.group(2))]
        patterns.append([parent] + sublabels)

    if patterns:
        return patterns

    # Simplified format
    simp_parent_re = re.compile(
        rf"<({parent_names_re})>(.*?)</(?:{parent_names_re})>",
        re.DOTALL | re.IGNORECASE,
    )
    simp_child_re = re.compile(r"<([a-zA-Z][^/>\s]*)(?:\s[^>]*)?>")

    for m in simp_parent_re.finditer(output_text):
        parent = m.group(1)
        sublabels = [
            c.group(1) for c in simp_child_re.finditer(m.group(2))
            if c.group(1) not in _PARENT_LABEL_NAMES
        ]
        patterns.append([parent] + sublabels)

    return patterns


# ---------------------------------------------------------------------------
# Internal helper — normalised sublabel pattern extraction
# ---------------------------------------------------------------------------

def _extract_label_pattern(output_text: str) -> dict[str, list[tuple]]:
    """
    Return normalised sublabel patterns for *output_text*.

    Calls each normalizer from ``patterns/normalizers.py`` on the raw text
    found under each (parent, child) path and returns the results as tuples
    (hashable, usable as dict keys or set members).

    Works with both full and simplified tag formats.
    """
    parents = _parse_parent_annotations(output_text)
    result: dict[str, list[tuple]] = {}

    for parent_label, child_label, normalizer in _SUBLABEL_PATHS:
        raw_strings = _get_sublabel_strings(parents, parent_label, child_label)
        result[f"{parent_label}_{child_label}"] = [
            tuple(normalizer(s)) for s in raw_strings
        ]

    return result


def _parse_parent_annotations(text: str) -> dict[str, list[dict]]:
    """
    Parse *text* into a dict keyed by parent label name.

    Handles both full (``<manual_label labelname="decision">``) and
    simplified (``<decision>``) formats.
    """
    soup = BeautifulSoup(text, "html.parser")
    result: dict[str, list[dict]] = {n: [] for n in _PARENT_LABEL_NAMES}

    # Full format
    for tag in soup.find_all(["manual_label", "auto_label"]):
        name = tag.get("labelname", "")
        if name in result:
            result[name].append(str(tag))

    if any(result.values()):
        return result

    # Simplified format - use regex for names with spaces (e.g., "secondary source")
    for name in _PARENT_LABEL_NAMES:
        if " " in name:
            # Use regex to find tags like <secondary source>...</secondary source>
            escaped_name = re.escape(name)
            tag_re = re.compile(rf"<{escaped_name}>(.*?)</{escaped_name}>", re.DOTALL | re.IGNORECASE)
            for m in tag_re.finditer(text):
                result[name].append(m.group(0))
        else:
            # For single-word names, use direct find_all
            for tag in soup.find_all(name):
                result[name].append(str(tag))

    return result


def _get_sublabel_strings(
    parents: dict[str, list[dict]],
    parent_label: str,
    child_label: str,
) -> list[str]:
    """
    Collect the text content of every *child_label* found inside *parent_label*
    annotations. Works with both full and simplified tag formats.
    """
    results: list[str] = []

    for ann in parents.get(parent_label, []):
        
        if not ann:
            continue
        soup = BeautifulSoup(ann, "html.parser")

        # Full format
        if "manual_label" in ann or "auto_label" in ann:
            elements = soup.find_all(
                ["manual_label", "auto_label"], attrs={"labelname": child_label}
            )
        else:
            # Simplified format
            elements = soup.find_all(child_label)

        for el in elements:
            text = el.get_text(strip=True)
            if text:
                results.append(text)

    return results


def get_list_of_mention(tokens, keep_labels, label_type=None):
    """
    Extract mentions from tokens and return their positions.
    
    Args:
        tokens: List of tokens to search
        keep_labels: List of label names to keep (e.g., ["title", "decision"])
        label_type: Optional filter - "manual_label", "auto_label", or None (both)
    
    Returns:
        List of tuples: (HTMLLabel object, start_index, end_index)
        - HTMLLabel object: The parsed opening tag
        - start_index: Index of the opening tag in tokens list
        - end_index: Index of the closing tag in tokens list
    """
    mentions = []
    i = 0
    
    while i < len(tokens):
        token = tokens[i]
        
        # Check if this matches the label type we're looking for
        is_match = False
        if label_type == "manual_label" and is_manual_label_tag(token) == 1:
            is_match = True
        elif label_type == "auto_label" and is_auto_label_tag(token) == 1:
            is_match = True
        elif label_type is None and (is_manual_label_tag(token) == 1 or is_auto_label_tag(token) == 1):
            is_match = True
        
        if is_match:
            html_label = HTMLLabel(token)
            
            # Check if this label is in keep_labels
            if html_label.name in keep_labels:
                start_index = i
                depth = 1
                i += 1
                
                # Find the matching closing tag
                while i < len(tokens) and depth > 0:
                    current_token = tokens[i]
                    
                    # Check if it's an opening tag of the same type
                    if label_type == "manual_label" and is_manual_label_tag(current_token) == 1:
                        depth += 1
                    elif label_type == "auto_label" and is_auto_label_tag(current_token) == 1:
                        depth += 1
                    elif label_type is None:
                        if is_manual_label_tag(current_token) == 1 or is_auto_label_tag(current_token) == 1:
                            depth += 1
                    
                    # Check if it's a closing tag of the same type
                    if label_type == "manual_label" and is_manual_label_tag(current_token) == 2:
                        depth -= 1
                    elif label_type == "auto_label" and is_auto_label_tag(current_token) == 2:
                        depth -= 1
                    elif label_type is None:
                        if is_manual_label_tag(current_token) == 2 or is_auto_label_tag(current_token) == 2:
                            depth -= 1
                    
                    if depth == 0:
                        end_index = i
                        mentions.append((html_label, start_index, end_index))
                        break
                    
                    i += 1
                continue
        
        i += 1
    
    return mentions

def build_processing_segments(tokens, parents):
    """
    Build a list of segments alternating between:
      - non-processable token spans
      - processable mention spans

    Returns:
        List[dict]: each dict has:
            - "process": bool
            - "tokens": list
            - "meta": optional mention metadata
    """
    segments = []
    cursor = 0

    for html_label, start_idx, end_idx in parents:
        # Non-processable tokens before the mention
        if cursor < start_idx:
            segments.append({
                "process": False,
                "tokens": tokens[cursor:start_idx]
            })

        # The mention itself (processable)
        segments.append({
            "process": True,
            "tokens": tokens[start_idx:end_idx + 1],
            "meta": {
                "label": html_label,
                "start": start_idx,
                "end": end_idx
            }
        })

        cursor = end_idx + 1

    # Trailing non-processable tokens
    if cursor < len(tokens):
        segments.append({
            "process": False,
            "tokens": tokens[cursor:]
        })

    return segments




# ---------------------------------------------------------------------------
# Internal helper — build a single example dict
# ---------------------------------------------------------------------------

def _build_example(
    input_text: str,
    output_text: str,
    source_file: str,
) -> dict:
    """
    Assemble one example dict from raw (input, output) strings.

    Applies spacing fix and computes both metadata fields.
    """
    input_text  = _fix_tag_spacing(input_text)
    output_text = _fix_tag_spacing(output_text)

    return {
        "example":            {"input": input_text, "output": output_text},
        "source_file":        source_file,
        "annotation_pattern": _extract_annotation_pattern(output_text),
        "label_pattern":      _extract_label_pattern(output_text),
    }





# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def extract_few_shot_examples(
    token_chunks: list[list[str]],
    cfg: LabelTransformConfig = None,
    source_file: str = "",
) -> list[dict]:
    """
    Build example dicts from a flat list of token chunks.

    The *input* is the chunk with all label tags stripped (plain text).
    The *output* is the chunk with labels transformed according to *cfg*.
    Spacing is fixed and both pattern fields are populated automatically.

    Args:
        token_chunks: List of token lists, one per chunk.
        cfg:          Label rendering / filtering options.
        source_file:  Original filename, stored in each example dict.

    Returns:
        List of example dicts.

    Examples:
        >>> cfg = LabelTransformConfig(use_simplified=True, remove_attributes=["style"])
        >>> examples = extract_few_shot_examples(chunks, cfg, source_file="doc1.html")

        >>> cfg = LabelTransformConfig(use_simplified=True, keep_labels=["decision"])
        >>> examples = extract_few_shot_examples(chunks, cfg)
    """
    examples: list[dict] = []

    for chunk in token_chunks:
        input_tokens = clean_tokens(
            chunk,
            normalize=True,
            keep_manual_label=False,
            keep_auto_label=False,
            keep_bookmarks=False,
        )
        if not cfg:
            output_tokens = chunk
        else:
            output_tokens = prepare_label_tokens(chunk, cfg)
        examples.append(
            _build_example(decode(input_tokens), decode(output_tokens), source_file)
        )

    print(f"   ✓ Extracted {len(examples)} few-shot examples from chunks")
    return examples