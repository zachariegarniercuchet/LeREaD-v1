"""
Utilities for generating and saving comparison plots.
"""
from pathlib import Path


def _ensure_ensemble_scores(log: list[dict], examples: list[dict], pattern_dict: dict = None) -> list[dict]:
    """
    Ensure each log entry has an ensemble_score.
    If missing, compute it from examples and their indices.
    """
    if log and 'ensemble_score' in log[0]:
        return log  # Already has scores
    
    # For random selection, compute scores on the fly
    if pattern_dict is None:
        pattern_dict = {}
    
    from src.fewshot.selector.greedy import _coverage_score, _KEY_MAP, _make_hashable
    
    score_lookup = {
        dict_key: {_make_hashable(pattern): score for score, pattern in entries}
        for dict_key, entries in pattern_dict.items()
    } if pattern_dict else {}
    
    enriched_log = []
    selected_patterns = []
    
    for entry in log:
        idx = entry['example_index']
        lp = examples[idx].get("label_pattern", {})
        selected_patterns.append(lp)
        
        ensemble_score, covered = _coverage_score(selected_patterns, score_lookup)
        enriched_entry = dict(entry)
        enriched_entry['ensemble_score'] = ensemble_score
        enriched_entry['n_patterns'] = len(covered)
        enriched_log.append(enriched_entry)
    
    return enriched_log


def plot_coverage_comparison(greedy_log, random_log, total_max=700, save_path="coverage_plot.png", examples=None, pattern_dict=None):
    """
    Generate a comparison plot of greedy vs random few-shot selection coverage.
    
    Args:
        greedy_log (list[dict]): Selection log from greedy_select_examples
        random_log (list[dict]): Selection log from random_select_examples
        total_max (float): Maximum possible score for normalization (default: 700)
        save_path (str | Path): Path where the plot will be saved
        examples (list[dict]): Required if random_log needs score computation
        pattern_dict (dict): Required if random_log needs score computation
    """
    import matplotlib.pyplot as plt
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure both logs have ensemble_score
    if examples and pattern_dict:
        greedy_log = _ensure_ensemble_scores(greedy_log, examples, pattern_dict)
        random_log = _ensure_ensemble_scores(random_log, examples, pattern_dict)
    
    fig, ax = plt.subplots(figsize=(10, 5))

    # Greedy curve
    greedy_steps    = [e['step'] for e in greedy_log]
    greedy_coverage = [e['ensemble_score'] / total_max * 100 for e in greedy_log]
    ax.plot(greedy_steps, greedy_coverage,
            marker='o', markersize=4, linewidth=2,
            color='steelblue', label='Greedy selection')

    # Random curve
    random_steps    = [e['step'] for e in random_log]
    random_coverage = [e['ensemble_score'] / total_max * 100 for e in random_log]
    ax.plot(random_steps, random_coverage,
            marker='o', markersize=4, linewidth=2,
            color='tomato', linestyle='--', label='Random selection')

    ax.set_xlabel("Number of Examples Selected", fontsize=13)
    ax.set_ylabel("Surface Coverage Score (%)", fontsize=13)
    ax.set_title("Greedy vs Random Coverage", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Plot saved to {save_path}")
