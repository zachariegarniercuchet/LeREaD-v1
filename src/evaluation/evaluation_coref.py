"""Coreference evaluation metrics (MUC, B-cubed, CEAFe, LEA) for the
LeREaD coref-resolution pipeline.

Mirrors evaluation_l1_util.py's role for the extraction pipeline: this is the
metrics engine, imported by evaluate_coref.py (the CLI) as well as usable
directly from a notebook.

Two ways to score:
  - evaluate_coref(gold, pred)            -> single-document P/R/F1 per metric
  - evaluate_coref_raw(gold, pred)         -> raw (num, den) counts per metric
    + aggregate_raw([...]) + finalize_raw(...) for corpus-level (micro)
    scoring across many documents — summing raw counts before dividing,
    rather than averaging per-document F1s, which is the standard
    CoNLL-scorer convention for coreference evaluation across a corpus.
"""

from coval.eval.evaluator import (
    muc,
    b_cubed,
    ceafe,
    lea,
)

METRICS = [("MUC", muc), ("B3", b_cubed), ("CEAF_e", ceafe), ("LEA", lea)]


def build_mention_to_gold_index(clusters):
    """mention -> cluster index (int), needed for muc/b_cubed/lea"""
    m2c = {}
    for i, cluster in enumerate(clusters):
        for m in cluster:
            m2c[m] = i
    return m2c


def evaluate_metric_raw(metric, pred_clusters, gold_clusters,
                         mention_to_gold, mention_to_pred):
    """Raw (p_num, p_den, r_num, r_den) counts for one metric — summable
    across documents for corpus-level aggregation."""
    if metric is ceafe:
        p_num, p_den, r_num, r_den = metric(pred_clusters, gold_clusters)
    elif metric is lea:
        p_num, p_den = metric(pred_clusters, gold_clusters, mention_to_gold)
        r_num, r_den = metric(gold_clusters, pred_clusters, mention_to_pred)
    else:  # muc, b_cubed
        p_num, p_den = metric(pred_clusters, mention_to_gold)
        r_num, r_den = metric(gold_clusters, mention_to_pred)

    return p_num, p_den, r_num, r_den


def evaluate_coref_raw(gold_clusters, pred_clusters):
    """Raw {metric_name: (p_num, p_den, r_num, r_den)} for one document."""
    gold_m2g = build_mention_to_gold_index(gold_clusters)
    pred_m2g = build_mention_to_gold_index(pred_clusters)

    raw = {}
    for name, metric in METRICS:
        raw[name] = evaluate_metric_raw(
            metric, pred_clusters, gold_clusters, gold_m2g, pred_m2g
        )
    return raw


def aggregate_raw(raw_list):
    """Sum raw counts across multiple documents (corpus-level / micro
    aggregation — the standard way to report coref metrics over a dataset)."""
    aggregated = {name: [0.0, 0.0, 0.0, 0.0] for name, _ in METRICS}
    for raw in raw_list:
        for name, _ in METRICS:
            p_num, p_den, r_num, r_den = raw[name]
            aggregated[name][0] += p_num
            aggregated[name][1] += p_den
            aggregated[name][2] += r_num
            aggregated[name][3] += r_den
    return {name: tuple(v) for name, v in aggregated.items()}


def finalize_raw(raw):
    """Turn raw (p_num, p_den, r_num, r_den) counts into precision/recall/f1
    per metric, plus the CoNLL average F1 (mean of MUC/B3/CEAF_e F1s —
    LEA is deliberately excluded from this average, following convention)."""
    scores = {}
    for name, _ in METRICS:
        p_num, p_den, r_num, r_den = raw[name]
        precision = p_num / p_den if p_den else 0.0
        recall = r_num / r_den if r_den else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        scores[name] = {"precision": precision, "recall": recall, "f1": f1}

    scores["CoNLL"] = {
        "f1": (scores["MUC"]["f1"] + scores["B3"]["f1"] + scores["CEAF_e"]["f1"]) / 3
    }
    return scores


def evaluate_coref(gold_clusters, pred_clusters):
    """Single-document P/R/F1 per metric + CoNLL average.
    Kept for backward compatibility with the dev notebook; internally this
    is just evaluate_coref_raw(...) -> finalize_raw(...)."""
    return finalize_raw(evaluate_coref_raw(gold_clusters, pred_clusters))


def print_evaluation_table(scores, title: str = "Coreference Evaluation") -> str:
    """Pretty-print precision/recall/F1 per metric as an aligned table, plus
    the CoNLL average F1. Returns the rendered string as well (so callers can
    capture it for a log file)."""
    metric_names = ["MUC", "B3", "CEAF_e", "LEA"]
    w_metric, w_val = 8, 10

    def row(cells):
        parts = [cells[0].ljust(w_metric)] + [c.ljust(w_val) for c in cells[1:]]
        return "│ " + " │ ".join(parts) + " │"

    top     = "┌" + "─"*(w_metric+2) + "┬" + "┬".join(["─"*(w_val+2)]*3) + "┐"
    midrule = "├" + "─"*(w_metric+2) + "┼" + "┼".join(["─"*(w_val+2)]*3) + "┤"
    bottom  = "└" + "─"*(w_metric+2) + "┴" + "┴".join(["─"*(w_val+2)]*3) + "┘"

    lines = [f"\n=== {title} ===", top, row(["Metric", "Precision", "Recall", "F1"]), midrule]
    for name in metric_names:
        s = scores[name]
        lines.append(row([name, f"{s['precision']:.4f}", f"{s['recall']:.4f}", f"{s['f1']:.4f}"]))
    lines.append(bottom)
    lines.append(f"CoNLL average F1 (MUC/B3/CEAF_e): {scores['CoNLL']['f1']:.4f}\n")

    rendered = "\n".join(lines)
    print(rendered)
    return rendered


def print_summary_table(rows, title: str = "Per-Document Summary") -> str:
    """Compact one-row-per-document table: F1 for each metric + CoNLL
    average. Meant for batch evaluation, where a full breakdown per document
    (see print_evaluation_table) would be too much to scan.

    Parameters
    ----------
    rows : list[tuple[str, dict]]
        (document_name, scores) pairs, where scores is the dict returned by
        finalize_raw (or evaluate_coref).
    """
    metric_names = ["MUC", "B3", "CEAF_e", "LEA"]
    name_w = max([len("Document")] + [len(name) for name, _ in rows])
    val_w = 8

    def row(cells):
        parts = [cells[0].ljust(name_w)] + [c.rjust(val_w) for c in cells[1:]]
        return "│ " + " │ ".join(parts) + " │"

    top     = "┌" + "─"*(name_w+2) + "┬" + "┬".join(["─"*(val_w+2)]*5) + "┐"
    midrule = "├" + "─"*(name_w+2) + "┼" + "┼".join(["─"*(val_w+2)]*5) + "┤"
    bottom  = "└" + "─"*(name_w+2) + "┴" + "┴".join(["─"*(val_w+2)]*5) + "┘"

    header_cells = ["Document"] + metric_names + ["CoNLL"]
    lines = [f"\n=== {title} ===", top, row(header_cells), midrule]
    for name, scores in rows:
        cells = [name] + [f"{scores[m]['f1']:.4f}" for m in metric_names] + [f"{scores['CoNLL']['f1']:.4f}"]
        lines.append(row(cells))
    lines.append(bottom)

    rendered = "\n".join(lines)
    print(rendered)
    return rendered