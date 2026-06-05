"""
LeREaD - Attribute Evaluation Module (Level 2)

Evaluates attribute-level agreement between gold-standard and system annotations,
for spans that already match at the span level (Level 1 exact match).

Methodology:
  1. Level 1 matching: exact match on (label, normalised text, normalised context).
  2. For each matched pair, collect all attributes except the excluded ones
     (style, parent, labelname, verified).
  3. Agreement for one attribute instance = 1 iff both values are identical
     after lower-casing and stripping whitespace.
  4. Accuracy = matching_attributes / total_attributes (micro-average).

Excluded attributes: style, parent, labelname, verified.

Compatible with the LeREaD span-level evaluation module (evaluate.py).
The Span class and extract_spans() are imported from there; only the
attribute logic lives here.
"""

import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Re-use span extraction + Level-1 matching from the companion module.
# Adjust the import path if evaluate.py is in a different location.
from .evaluation_l1_util import Span, extract_spans, CONTEXT_CHARS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXCLUDED_ATTRS = {"style", "parent", "labelname", "verified"}


# ---------------------------------------------------------------------------
# Level-1 matching (greedy, exact)
# ---------------------------------------------------------------------------

def _match_spans_l1(
    gold: List[Span], system: List[Span]
) -> List[Tuple[Span, Span]]:
    """
    Return a list of (gold_span, system_span) pairs that match at Level 1
    (same label + same normalised text + same normalised context).
    Each span is used at most once.
    """
    used = [False] * len(system)
    pairs: List[Tuple[Span, Span]] = []

    for g in gold:
        for j, s in enumerate(system):
            if not used[j] and g.matches(s):
                pairs.append((g, s))
                used[j] = True
                break

    return pairs


# ---------------------------------------------------------------------------
# Attribute extraction helpers
# ---------------------------------------------------------------------------

def _get_attributes(span: Span) -> Dict[str, str]:
    """Return span attributes with excluded keys removed."""
    return {
        k: v
        for k, v in span.attributes.items()
        if k not in EXCLUDED_ATTRS
    }


def _attr_agree(v1: Optional[str], v2: Optional[str]) -> bool:
    """True iff both values are non-None and equal after normalisation."""
    if v1 is None or v2 is None:
        return False
    return v1.lower().strip() == v2.lower().strip()


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_attribute_metrics(
    gold: List[Span],
    system: List[Span],
) -> Tuple[Dict, Dict[str, Dict], Dict[str, Dict], Dict[str, Dict[str, Dict]]]:
    """
    Compute Level-2 attribute agreement for one (gold, system) pair.

    Parameters
    ----------
    gold : List[Span]
        Spans extracted from the gold file.
    system : List[Span]
        Spans extracted from the system file.

    Returns
    -------
    overall : dict
        Aggregate counts and accuracy across all matched spans.
    per_label : dict
        Per-label breakdown  {label -> {matched_spans, total, matching, accuracy,
                                        attributes: {attr -> {total, matching, accuracy}}}}.
    overall_attrs : dict
        Per-attribute breakdown collapsed across labels
        {attr -> {total, matching, accuracy}}.
    category_attrs : dict
        Per-attribute breakdown split by top-level category (legislation /
        decision / secondary sources / other)
        {category -> {attr -> {total, matching, accuracy}}}.
    """
    matched_pairs = _match_spans_l1(gold, system)

    if not matched_pairs:
        return (
            {"matched_spans": 0, "total_attributes": 0,
             "matching_attributes": 0, "accuracy": 0.0},
            {}, {}, {},
        )

    # --- accumulators ---
    total_attrs    = 0
    matching_attrs = 0

    per_label_acc: Dict[str, dict] = defaultdict(lambda: {
        "spans": 0, "total": 0, "matching": 0,
        "attributes": defaultdict(lambda: {"total": 0, "matching": 0}),
    })

    overall_attr_acc: Dict[str, dict] = defaultdict(
        lambda: {"total": 0, "matching": 0}
    )

    category_attr_acc: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "matching": 0})
    )

    for g, s in matched_pairs:
        # ---- determine category -----------------------------------------
        lname = g.labelname.lower()
        if lname in {"legislation", "decision", "secondary sources"}:
            category = lname
        else:
            parent = g.attributes.get("parent", "")
            top    = parent.split(",")[0].strip().lower() if parent else ""
            category = (
                top if top in {"legislation", "decision", "secondary sources"}
                else "other"
            )

        # ---- attribute comparison ----------------------------------------
        attrs_g = _get_attributes(g)
        attrs_s = _get_attributes(s)
        all_keys = set(attrs_g) | set(attrs_s)

        per_label_acc[g.labelname]["spans"] += 1

        for key in all_keys:
            v_g = attrs_g.get(key)
            v_s = attrs_s.get(key)
            agree = int(_attr_agree(v_g, v_s))

            total_attrs    += 1
            matching_attrs += agree

            per_label_acc[g.labelname]["total"]   += 1
            per_label_acc[g.labelname]["matching"] += agree
            per_label_acc[g.labelname]["attributes"][key]["total"]    += 1
            per_label_acc[g.labelname]["attributes"][key]["matching"] += agree

            overall_attr_acc[key]["total"]    += 1
            overall_attr_acc[key]["matching"] += agree

            category_attr_acc[category][key]["total"]    += 1
            category_attr_acc[category][key]["matching"] += agree

    # --- build output dicts ---
    accuracy = matching_attrs / total_attrs if total_attrs else 0.0

    overall = {
        "matched_spans":     len(matched_pairs),
        "total_attributes":  total_attrs,
        "matching_attributes": matching_attrs,
        "accuracy":          accuracy,
    }

    per_label: Dict[str, Dict] = {}
    for label, acc in per_label_acc.items():
        lbl_acc = acc["matching"] / acc["total"] if acc["total"] else 0.0
        per_label[label] = {
            "matched_spans":      acc["spans"],
            "total_attributes":   acc["total"],
            "matching_attributes": acc["matching"],
            "accuracy":           lbl_acc,
            "attributes": {
                attr: {
                    "total":    ad["total"],
                    "matching": ad["matching"],
                    "accuracy": ad["matching"] / ad["total"] if ad["total"] else 0.0,
                }
                for attr, ad in acc["attributes"].items()
            },
        }

    overall_attrs: Dict[str, Dict] = {
        attr: {
            "total":    ad["total"],
            "matching": ad["matching"],
            "accuracy": ad["matching"] / ad["total"] if ad["total"] else 0.0,
        }
        for attr, ad in overall_attr_acc.items()
    }

    category_attrs: Dict[str, Dict[str, Dict]] = {
        cat: {
            attr: {
                "total":    ad["total"],
                "matching": ad["matching"],
                "accuracy": ad["matching"] / ad["total"] if ad["total"] else 0.0,
            }
            for attr, ad in cat_attrs.items()
        }
        for cat, cat_attrs in category_attr_acc.items()
    }

    return overall, per_label, overall_attrs, category_attrs


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_SEP_WIDE  = "─" * 110
_SEP_DWIDE = "═" * 110


def _fmt_cell(data: Optional[Dict]) -> str:
    """Format one category cell as 'match/total (XX.X%)'."""
    if data is None:
        return "-"
    return f"{data['matching']}/{data['total']} ({data['accuracy']*100:.1f}%)"


def _print_attribute_table(
    overall_attrs: Dict[str, Dict],
    category_attrs: Dict[str, Dict[str, Dict]],
) -> None:
    """Print the per-attribute × per-category accuracy table."""
    COL = 22
    print(f"\n{'Attribute':<20} {'Legislation':<{COL}} {'Decision':<{COL}} "
          f"{'Secondary Sources':<{COL}} {'Total':>6} {'Match':>6} {'Accuracy':>9}")
    print(_SEP_WIDE)

    for attr in sorted(overall_attrs):
        leg = category_attrs.get("legislation",        {}).get(attr)
        dec = category_attrs.get("decision",           {}).get(attr)
        sec = category_attrs.get("secondary sources",  {}).get(attr)
        od  = overall_attrs[attr]

        print(
            f"{attr:<20} {_fmt_cell(leg):<{COL}} {_fmt_cell(dec):<{COL}} "
            f"{_fmt_cell(sec):<{COL}} "
            f"{od['total']:>6} {od['matching']:>6} {od['accuracy']*100:>8.1f}%"
        )


def print_results(
    overall: Dict,
    per_label: Dict[str, Dict],
    overall_attrs: Dict[str, Dict],
    category_attrs: Dict[str, Dict[str, Dict]],
    gold_name: str = "",
    system_name: str = "",
) -> None:
    """
    Print Level-2 attribute evaluation results.

    Parameters
    ----------
    overall, per_label, overall_attrs, category_attrs
        As returned by compute_attribute_metrics() or evaluate_attribute().
    gold_name, system_name : str
        Display names (file basenames) for the header.
    """
    # ── header ──────────────────────────────────────────────────────────────
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " LEVEL 2 – ATTRIBUTE ACCURACY EVALUATION".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    if gold_name:
        print(f"\n  Gold  : {gold_name}")
    if system_name:
        print(f"  System: {system_name}")

    print(f"\n  Excluded attributes : {', '.join(sorted(EXCLUDED_ATTRS))}")
    print( "  Agreement criterion : exact match (case-insensitive, stripped)")

    # ── overall ─────────────────────────────────────────────────────────────
    print(f"\n{_SEP_WIDE}")
    print("  OVERALL ATTRIBUTE ACCURACY")
    print(_SEP_WIDE)
    print(f"\n  Matched Spans (Level 1)     : {overall['matched_spans']:>5}")
    print(f"  Attribute instances compared: {overall['total_attributes']:>5}")
    print(f"  Matching attribute values   : {overall['matching_attributes']:>5}")
    print(f"  Accuracy                    : {overall['accuracy']*100:>8.2f}%")

    # ── per-label ────────────────────────────────────────────────────────────
    if per_label:
        print(f"\n{_SEP_WIDE}")
        print("  PER-LABEL BREAKDOWN")
        print(_SEP_WIDE)

        W = 36
        print(f"\n  {'Label':<{W}} {'Spans':>6} {'Total':>6} {'Match':>6} {'Accuracy':>9}")
        print("  " + "─" * 65)

        for label in sorted(per_label):
            m = per_label[label]
            print(f"  {label:<{W}} {m['matched_spans']:>6} "
                  f"{m['total_attributes']:>6} {m['matching_attributes']:>6} "
                  f"{m['accuracy']*100:>8.1f}%")

            # per-attribute detail under each label (indented)
            for attr in sorted(m["attributes"]):
                ad = m["attributes"][attr]
                print(f"    {'↳ '+attr:<{W}} {'':>6} "
                      f"{ad['total']:>6} {ad['matching']:>6} "
                      f"{ad['accuracy']*100:>8.1f}%")

    # ── per-attribute × category ─────────────────────────────────────────────
    if overall_attrs:
        print(f"\n{_SEP_WIDE}")
        print("  ATTRIBUTE ACCURACY BY CATEGORY")
        print(_SEP_WIDE)
        _print_attribute_table(overall_attrs, category_attrs)

    print(f"\n{_SEP_DWIDE}\n")


# ---------------------------------------------------------------------------
# Single-pair evaluation
# ---------------------------------------------------------------------------

def evaluate_attribute(
    gold_path: str,
    system_path: str,
    context_chars: int = CONTEXT_CHARS,
    verbose: bool = True,
) -> Tuple[Dict, Dict[str, Dict], Dict[str, Dict], Dict[str, Dict[str, Dict]]]:
    """
    Level-2 attribute evaluation for one (gold, system) HTML file pair.

    Parameters
    ----------
    gold_path : str
        Path to the gold-standard HTML annotation file.
    system_path : str
        Path to the system-generated HTML annotation file.
    context_chars : int
        Context window (characters) used for Level-1 span matching.
        Must match the value used in the Level-1 evaluation for consistency.
    verbose : bool
        If True, print a formatted results table.

    Returns
    -------
    (overall, per_label, overall_attrs, category_attrs)
        See compute_attribute_metrics() for field descriptions.
    """
    gold_spans   = extract_spans(gold_path,   context_chars)
    system_spans = extract_spans(system_path, context_chars)

    overall, per_label, overall_attrs, category_attrs = compute_attribute_metrics(
        gold_spans, system_spans
    )

    if verbose:
        print_results(
            overall, per_label, overall_attrs, category_attrs,
            gold_name=Path(gold_path).name,
            system_name=Path(system_path).name,
        )

    return overall, per_label, overall_attrs, category_attrs


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_attribute_batch(
    pairs: List[Tuple[str, str]],
    context_chars: int = CONTEXT_CHARS,
    verbose_per_file: bool = False,
) -> Tuple[Dict, Dict[str, Dict], Dict[str, Dict], Dict[str, Dict[str, Dict]]]:
    """
    Level-2 attribute evaluation across a list of (gold, system) file pairs.

    Micro-averaging: raw counts (total / matching attribute instances) are
    summed across all documents per label / attribute, then accuracy is
    computed from those sums.  Each attribute instance counts equally.

    Parameters
    ----------
    pairs : list of (gold_path, system_path)
    context_chars : int
        Context window used for Level-1 span matching.
    verbose_per_file : bool
        If True, print a full results table for each file pair.

    Returns
    -------
    (overall, per_label, overall_attrs, category_attrs)
        Micro-averaged results with the same structure as evaluate_attribute().
    """
    # Accumulators (raw counts)
    acc_overall: Dict[str, int] = defaultdict(int)
    acc_per_label: Dict[str, dict] = defaultdict(lambda: {
        "spans": 0, "total": 0, "matching": 0,
        "attributes": defaultdict(lambda: {"total": 0, "matching": 0}),
    })
    acc_overall_attrs: Dict[str, dict] = defaultdict(
        lambda: {"total": 0, "matching": 0}
    )
    acc_category_attrs: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "matching": 0})
    )

    per_file_accuracy: List[float] = []   # for macro reporting
    n_processed = 0

    print(f"\nLevel-2 batch evaluation — {len(pairs)} file pair(s)")
    print("=" * 70)

    for i, (gold_path, system_path) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] {Path(gold_path).name}")

        try:
            gold_spans   = extract_spans(gold_path,   context_chars)
            system_spans = extract_spans(system_path, context_chars)
        except Exception as exc:
            print(f"  ⚠ Skipped — {exc}")
            continue

        overall, per_label, overall_attrs, category_attrs = \
            compute_attribute_metrics(gold_spans, system_spans)

        if verbose_per_file:
            print_results(
                overall, per_label, overall_attrs, category_attrs,
                gold_name=Path(gold_path).name,
                system_name=Path(system_path).name,
            )
        else:
            acc = overall["accuracy"]
            print(f"  matched spans: {overall['matched_spans']:>4}  |  "
                  f"attribute instances: {overall['total_attributes']:>5}  |  "
                  f"accuracy: {acc*100:.1f}%")

        # Accumulate overall counts
        acc_overall["matched_spans"]       += overall["matched_spans"]
        acc_overall["total_attributes"]    += overall["total_attributes"]
        acc_overall["matching_attributes"] += overall["matching_attributes"]

        # Accumulate per-label
        for label, m in per_label.items():
            acc_per_label[label]["spans"]   += m["matched_spans"]
            acc_per_label[label]["total"]   += m["total_attributes"]
            acc_per_label[label]["matching"] += m["matching_attributes"]
            for attr, ad in m["attributes"].items():
                acc_per_label[label]["attributes"][attr]["total"]    += ad["total"]
                acc_per_label[label]["attributes"][attr]["matching"] += ad["matching"]

        # Accumulate overall attrs
        for attr, ad in overall_attrs.items():
            acc_overall_attrs[attr]["total"]    += ad["total"]
            acc_overall_attrs[attr]["matching"] += ad["matching"]

        # Accumulate category attrs
        for cat, attrs in category_attrs.items():
            for attr, ad in attrs.items():
                acc_category_attrs[cat][attr]["total"]    += ad["total"]
                acc_category_attrs[cat][attr]["matching"] += ad["matching"]

        if overall["total_attributes"] > 0:
            per_file_accuracy.append(overall["accuracy"])

        n_processed += 1

    if n_processed == 0:
        print("\n  No files could be processed.")
        return (
            {"matched_spans": 0, "total_attributes": 0,
             "matching_attributes": 0, "accuracy": 0.0},
            {}, {}, {},
        )

    # ── build micro-averaged results ─────────────────────────────────────────
    tot   = acc_overall["total_attributes"]
    match = acc_overall["matching_attributes"]

    final_overall = {
        "matched_spans":       acc_overall["matched_spans"],
        "total_attributes":    tot,
        "matching_attributes": match,
        "accuracy":            match / tot if tot else 0.0,
    }

    final_per_label: Dict[str, Dict] = {}
    for label, acc in acc_per_label.items():
        lbl_acc = acc["matching"] / acc["total"] if acc["total"] else 0.0
        final_per_label[label] = {
            "matched_spans":       acc["spans"],
            "total_attributes":    acc["total"],
            "matching_attributes": acc["matching"],
            "accuracy":            lbl_acc,
            "attributes": {
                attr: {
                    "total":    ad["total"],
                    "matching": ad["matching"],
                    "accuracy": ad["matching"] / ad["total"] if ad["total"] else 0.0,
                }
                for attr, ad in acc["attributes"].items()
            },
        }

    final_overall_attrs: Dict[str, Dict] = {
        attr: {
            "total":    ad["total"],
            "matching": ad["matching"],
            "accuracy": ad["matching"] / ad["total"] if ad["total"] else 0.0,
        }
        for attr, ad in acc_overall_attrs.items()
    }

    final_category_attrs: Dict[str, Dict[str, Dict]] = {
        cat: {
            attr: {
                "total":    ad["total"],
                "matching": ad["matching"],
                "accuracy": ad["matching"] / ad["total"] if ad["total"] else 0.0,
            }
            for attr, ad in cat_attrs.items()
        }
        for cat, cat_attrs in acc_category_attrs.items()
    }

    # ── print batch summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print_results(
        final_overall, final_per_label, final_overall_attrs, final_category_attrs,
        gold_name="",
        system_name="",
    )

    macro_acc = (
        sum(per_file_accuracy) / len(per_file_accuracy)
        if per_file_accuracy else 0.0
    )
    print(f"  Macro accuracy (mean per-document accuracy): {macro_acc*100:.2f}%\n")

    return final_overall, final_per_label, final_overall_attrs, final_category_attrs


# ---------------------------------------------------------------------------
# Entry point — edit the paths below to run
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Single pair ──────────────────────────────────────────────────────────
    GOLD   = "data/annotated/test/my_document.html"
    SYSTEM = "data/supplementary/llm_extraction/my_document.html"

    evaluate_attribute(GOLD, SYSTEM)

    # ── Batch — Option A: list pairs manually ────────────────────────────────
    pairs = [
        ("data/annotated/test/doc1.html", "data/supplementary/llm_extraction/doc1.html"),
        ("data/annotated/test/doc2.html", "data/supplementary/llm_extraction/doc2.html"),
        # …
    ]

    evaluate_attribute_batch(pairs)

    # ── Batch — Option B: auto-match by filename ─────────────────────────────
    # from pathlib import Path
    # GOLD_DIR   = Path("data/annotated/test")
    # SYSTEM_DIR = Path("data/supplementary/llm_extraction")
    # pairs = [
    #     (str(g), str(SYSTEM_DIR / g.name))
    #     for g in sorted(GOLD_DIR.glob("*.html"))
    #     if (SYSTEM_DIR / g.name).exists()
    # ]
    # evaluate_attribute_batch(pairs)