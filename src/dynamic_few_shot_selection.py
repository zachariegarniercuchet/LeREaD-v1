"""
─────────────────────
Combines:
  • n_general  fixed/random examples (same logic as main_few_shot_selection)
  • n_dynamic  greedy-coverage examples driven by the chunk's pattern metadata
"""

project_root = r"."

import json
import random
from typing import Optional

from .tokenizer_utils import decode, tokenize
from .few_shot_utils import prepare_label_tokens

from .normalization import (
    normalize_fragment,
    normalize_decision_citation,
    normalize_legislation_citation,
    normalize_secondary_source,
    normalize_decision_title,
)

with open(fr"{project_root}\few_shot_selection_tool\pattern_dicts.json", 'r', encoding='utf-8') as f:
    pattern_dicts = json.load(f)

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _simplify_output(raw_output, label_config):
    """Re-uses your existing decode/prepare_label_tokens/tokenize pipeline."""
    return decode(prepare_label_tokens(tokenize(raw_output), label_config=label_config))


def _coverage_set(label_pattern: dict) -> set[tuple]:
    """
    Converts a label_pattern dict from a JSON example into a set of
    (category, tuple(pattern)) pairs for O(1) lookup.

    label_pattern = {
        "decision_fragment": [["PARA", "NUM"], ...],
        "legislation_fragment": [],
        ...
    }
    → {("decision_fragment", ("PARA", "NUM")), ...}
    """
    covered = set()
    for category, patterns in label_pattern.items():
        for pat in patterns:
            covered.add((category, tuple(pat)))
    return covered


def _metadata_to_needs(metadata: dict) -> set[tuple]:
    """
    Converts extract_pattern_metadata() output into the same
    (category, tuple(pattern)) format.

    metadata = {
        "decision_fragment": [[6.96, ["NUM"]], [0.23, ["NUM","AND","NUM"]]],
        ...
    }
    → {("decision_fragment", ("NUM",)), ("decision_fragment", ("NUM","AND","NUM")), ...}
    """
    needs = set()
    for category, scored_patterns in metadata.items():
        for _score, pat in scored_patterns:
            needs.add((category, tuple(pat)))
    return needs


def _mask_same_doc(items: list[dict], filename: str) -> list[dict]:
    return [item for item in items if filename not in item.get("source_file", "")]


def _extract_example(item: dict, label_config) -> dict:
    """Returns a ready-to-use dict with input/output/source_file/coverage."""
    ex = item.get("example", {})
    return {
        "input":       ex.get("input", ""),
        "output":      ex.get("output", ""),
        "source_file": item.get("source_file", ""),
        "coverage":    _coverage_set(item.get("label_pattern", {})),
    }

def _contains_pattern(token_sequence: list[str], pattern: list[str]) -> bool:
    """
    Sliding-window search: returns True if `pattern` appears as a
    consecutive subsequence anywhere inside `token_sequence`.
    """
    n, p = len(token_sequence), len(pattern)
    if p == 0 or p > n:
        return False
    for i in range(n - p + 1):
        if token_sequence[i : i + p] == pattern:
            return True
    return False

def find_matching_patterns(
    token_sequence: list[str],
    pattern_list: list[list],
) -> list[list]:
    """
    Returns all [score, pattern] entries from `pattern_list` whose
    pattern is found as a consecutive subsequence in `token_sequence`.
    """
    return [
        entry                          # [score, pattern]
        for entry in pattern_list
        if _contains_pattern(token_sequence, entry[1])
    ]
 
 
# ── 4. Main metadata builder ──────────────────────────────────────────────────
 
def extract_pattern_metadata(text: str) -> dict[str, list[list]]:
    """
    Tokenizes `text` with every normalization function, checks each
    token sequence against the relevant pattern dictionaries, and
    returns a metadata dict containing only the categories where at
    least one pattern matched.
 
    Parameters
    ----------
    text : str
        A single text chunk (e.g. the first chunk of a document).
 
    Returns
    -------
    dict
        Keys are category names (e.g. "decision_citation").
        Values are lists of [score, pattern] for every matched pattern.
        Categories with zero matches are omitted.
 
    Example output
    --------------
    {
        "decision_fragment":    [[49.3, ["PARA", "NUM"]]],
        "legislation_fragment": [[49.3, ["PARA", "NUM"]], [16.36, ["P", "NUM"]]],
        "decision_citation":    [[49.3, ["PARA", "NUM"]]],
    }
    """
 
    # ── 4a. Tokenize once per normalization function ──────────────────────────
    tok_fragment          = normalize_fragment(text)
    tok_decision_citation = normalize_decision_citation(text)
    tok_legislation_citation = normalize_legislation_citation(text)
    tok_secondary_source  = normalize_secondary_source(text)
    tok_decision_title    = normalize_decision_title(text)
 
    # ── 4b. Map each (token_sequence, category) pair ─────────────────────────
    # normalize_fragment feeds THREE categories (same tokens, different dicts)
    checks: list[tuple[list[str], str]] = [
        (tok_fragment,             "decision_fragment"),
        (tok_fragment,             "legislation_fragment"),
        (tok_fragment,             "sec_sources_fragment"),
        (tok_decision_citation,    "decision_citation"),
        (tok_legislation_citation, "legislation_citation"),
        (tok_secondary_source,     "sec_sources_source"),
        (tok_decision_title,       "decision_title"),
    ]
 
    # ── 4c. Run matching and collect results ──────────────────────────────────
    metadata: dict[str, list[list]] = {}
 
    for token_sequence, category in checks:
        matches = find_matching_patterns(token_sequence, pattern_dicts[category])
        if matches:
            metadata[category] = matches
 
    return metadata


# ═════════════════════════════════════════════════════════════════════════════
# GENERAL SELECTION  (mirrors main_few_shot_selection)
# ═════════════════════════════════════════════════════════════════════════════

def _select_general(
    fs_data:       list[dict],
    filename:      str,
    n_general:     int,
    label_config,
    use_random:    bool,
    rng:           random.Random,
) -> tuple[list[tuple], set[tuple]]:
    """
    Returns:
      selected_examples  – list of (input, simplified_output)
      covered_patterns   – set of (category, pattern_tuple) already covered
    """
    masked = _mask_same_doc(fs_data, filename)

    if use_random:
        prob = n_general / max(len(masked), 1)
        pool = [item for item in masked if rng.random() <= prob]
    else:
        pool = [item for item in masked if item.get("selected", False)]

    pool = pool[:n_general]

    examples   = []
    covered    = set()
    for item in pool:
        ex = _extract_example(item, label_config)
        simplified = _simplify_output(ex["output"], label_config)
        examples.append((ex["input"], simplified))
        covered |= ex["coverage"]

    return examples, covered


# ═════════════════════════════════════════════════════════════════════════════
# DYNAMIC SELECTION  (greedy set-cover)
# ═════════════════════════════════════════════════════════════════════════════

def _select_dynamic(
    dyn_data:         list[dict],
    filename:         str,
    n_dynamic:        int,
    label_config,
    already_covered:  set[tuple],
    chunk_needs:      set[tuple],
    rng:              random.Random,
) -> list[tuple]:
    """
    Greedy set-cover over the patterns still needed after general selection.
    At each step picks the candidate that covers the most remaining patterns,
    with random tie-breaking.  Stops when n_dynamic examples are chosen or
    all patterns are covered.
    """
    remaining = chunk_needs - already_covered
    if not remaining or n_dynamic == 0:
        return []

    # Build candidate pool (mask same doc, must cover ≥1 needed pattern)
    candidates = []
    for item in _mask_same_doc(dyn_data, filename):
        ex       = _extract_example(item, label_config)
        relevant = ex["coverage"] & remaining          # patterns it can cover
        if relevant:
            candidates.append({
                "input":    ex["input"],
                "output":   ex["output"],
                "coverage": ex["coverage"],
                "relevant": relevant,
            })

    selected   = []
    covered    = set(already_covered)                  # local copy

    while candidates and len(selected) < n_dynamic and remaining:

        # Score each candidate by how many *still* uncovered patterns it adds
        best_gain = max(len(c["coverage"] & remaining) for c in candidates)

        # All candidates tied at best_gain → random tie-break
        top = [c for c in candidates if len(c["coverage"] & remaining) == best_gain]
        chosen = rng.choice(top)

        simplified = _simplify_output(chosen["output"], label_config)
        selected.append((chosen["input"], simplified))

        # Update remaining
        covered   |= chosen["coverage"]
        remaining  = chunk_needs - covered

        # Remove chosen + any candidate now fully redundant (gain == 0)
        candidates = [
            c for c in candidates
            if c is not chosen and len(c["coverage"] & remaining) > 0
        ]

    return selected


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

def select_few_shot_for_chunk(
    chunk_metadata:   dict,
    filename:         str,
    label_config,
    general_json_path: str,
    dynamic_json_path: str,
    n_general:        int                = 20,
    n_dynamic:        int                = 10,
    
    random_seed:      Optional[int]     = None,
) -> list[tuple]:
    """
    Build a few-shot example list for a single chunk.

    Parameters
    ----------
    chunk_metadata   : output of extract_pattern_metadata(text_chunk)
    filename         : base name of the document being annotated (mask filter)
    label_config     : passed through to decode/prepare_label_tokens/tokenize
    n_general        : max examples from the general/fixed pool
    n_dynamic        : max additional examples chosen by greedy pattern coverage
    general_json_path: path to the general few-shot JSON
                       (defaults to combined_v3_with_sources_manual_label.json)
    dynamic_json_path: path to the dynamic/coverage JSON
                       (defaults to greedy_set_coverage_rejected_corrected.json)
    random_seed      : if set, enables random general selection mode

    Returns
    -------
    list of (input_text, simplified_output) tuples,
    length ≤ n_general + n_dynamic.
    """
    use_random = random_seed is not None
    rng        = random.Random(random_seed)


    # ── load data ─────────────────────────────────────────────────────────────
    general_data = _load_json(general_json_path)
    dynamic_data = _load_json(dynamic_json_path)

    #print(f"   ✓ General pool : {len(general_data)} examples from {general_json_path}")
    #print(f"   ✓ Dynamic pool : {len(dynamic_data)} examples from {dynamic_json_path}")
    #print(f"   ✓ Mode         : {'random' if use_random else 'manual'}")

    # ── convert chunk metadata to needed (category, pattern) pairs ────────────
    chunk_needs = _metadata_to_needs(chunk_metadata)
    #print(f"   ✓ Patterns needed by chunk : {len(chunk_needs)}")

    # ── step 1: general selection ─────────────────────────────────────────────
    general_examples, already_covered = _select_general(
        general_data, filename, n_general, label_config, use_random, rng
    )
    #print(f"   ✓ General examples selected : {len(general_examples)}")
    #print(f"   ✓ Patterns already covered  : {len(already_covered & chunk_needs)}"
    #      f" / {len(chunk_needs)}")

    # ── step 2: dynamic greedy selection ──────────────────────────────────────
    dynamic_examples = _select_dynamic(
        dynamic_data, filename, n_dynamic, label_config,
        already_covered, chunk_needs, rng
    )
    #print(f"   ✓ Dynamic examples selected : {len(dynamic_examples)}")

    # ── combine & return ──────────────────────────────────────────────────────
    all_examples = general_examples + dynamic_examples
    #print(f"   ✓ Total few-shot examples   : {len(all_examples)}")
    return all_examples


# ═════════════════════════════════════════════════════════════════════════════
# BATCH HELPER  (iterate over all chunks)
# ═════════════════════════════════════════════════════════════════════════════

def select_few_shot_for_all_chunks(
    token_chunks_par: list,
    filename:         str,
    label_config,
    n_general:        int             = 20,
    n_dynamic:        int             = 10,
    general_json_path: Optional[str] = None,
    dynamic_json_path: Optional[str] = None,
    random_seed:      Optional[int]  = None,
) -> list[list[tuple]]:
    """
    Runs select_few_shot_for_chunk() for every chunk.

    Parameters
    ----------
    token_chunks_par : list of token sequences (your existing variable)
    All other params  : forwarded to select_few_shot_for_chunk()

    Returns
    -------
    List (one entry per chunk) of few-shot example lists.
    """
    all_chunk_few_shots = []

    for i, chunk in enumerate(token_chunks_par):
        text_chunk = decode(chunk)
        metadata   = extract_pattern_metadata(text_chunk)

        #print(f"\n── Chunk {i+1}/{len(token_chunks_par)} ──────────────────────────")
        few_shots = select_few_shot_for_chunk(
            chunk_metadata    = metadata,
            filename          = filename,
            label_config      = label_config,
            n_general         = n_general,
            n_dynamic         = n_dynamic,
            general_json_path = general_json_path,
            dynamic_json_path = dynamic_json_path,
            random_seed       = random_seed,
        )
        all_chunk_few_shots.append(few_shots)

    return all_chunk_few_shots


# ═════════════════════════════════════════════════════════════════════════════
# QUICK SMOKE-TEST
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Fake metadata (replace with extract_pattern_metadata(text_chunk))
    fake_metadata = {
        "decision_fragment":   [[6.96, ["NUM"]], [0.23, ["NUM", "AND", "NUM"]]],
        "legislation_fragment": [[49.3, ["PARA", "NUM"]]],
    }

    few_shots = select_few_shot_for_chunk(
        chunk_metadata = fake_metadata,
        filename       = "my_document",
        label_config   = None,           # replace with your real label_config
        n_general      = 20,
        n_dynamic      = 10,
        random_seed    = 42,
    )

    print(f"\nFinal few-shot count: {len(few_shots)}")
    for inp, out in few_shots[:2]:
        print(f"\nINPUT : {inp[:80]}...")
        print(f"OUTPUT: {out[:80]}...")