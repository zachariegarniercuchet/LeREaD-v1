"""Token-level operations for merging and manipulating token sequences."""


import itertools


def flatten_token_chunks(token_chunks: list[list[str]], separator: str = None) -> list[str]:
    """
    Flatten a list of token chunks (list of lists) back into a single token list.
    Preserves token order exactly.
    
    Args:
        token_chunks: List of token chunks to flatten
        separator: Optional separator token to insert between chunks (e.g., '<sep>')
    """
    if separator:
        flat = []
        for i, chunk in enumerate(token_chunks):
            flat.extend(chunk)
            if i < len(token_chunks) - 1:  # Don't add separator after last chunk
                flat.append(separator)
    else:
        flat = list(itertools.chain.from_iterable(token_chunks))
    
    print(f"   ✓ Flattened {len(token_chunks)} chunks into {len(flat)} tokens")
    return flat


def merge_tokens_general(original_tokens: list[str], 
                        derived_tokens: list[str], 
                        is_protected_func,
                        is_opening_protected_func=None,
                        is_tag_token_func=None,
                        log: bool = False) -> list[str]:
    """
    GENERALIZED VERSION: Merge original tokens with derived tokens.
    
    Goal: Produce the original text with protected tokens (e.g., <sep>, <auto_label>) 
    inserted from the derived version.
    
    Algorithm:
    - If tokens are equivalent (same or both whitespace): take original token, advance both
    - If tokens differ:
      - If derived token is a CLOSING protected tag: emit it directly, advance idx2 only
      - If derived token is an OPENING protected tag:
          * Look ahead in derived for first non-tag word after the opening tag
          * Find that word in original (searching forward from idx1)
          * Flush all original tokens up to (but not including) that word
          * Emit the opening tag, advance idx2 only
          * (Handles cases like <i> appearing in original before the target word)
      - If derived token is a non-protected mismatch:
          * Try merging consecutive original tokens to match derived token
          * If merged: emit originals, advance idx1 by merge_count, idx2 by 1
          * Otherwise: emit t1, advance idx1 only
    
    Args:
        original_tokens: Original token list (without protected tokens)
        derived_tokens: Derived token list (with protected tokens inserted)
        is_protected_func: Returns True if a token is protected (opening or closing)
        is_opening_protected_func: Returns True if a token is an OPENING protected tag.
                                   If None, all protected tokens are treated as atomic
                                   insertions (old behaviour — no lookahead).
        is_tag_token_func: Returns True if a token is any kind of tag (used to skip
                           non-content tokens while searching for the anchor word).
                           Only used when is_opening_protected_func is provided.
                           If None, defaults to: lambda t: t.startswith('<') and t.endswith('>')
        log: Print debug information
    
    Returns:
        Merged token list with original tokens + protected tokens from derived
    
    Example (with opening-tag lookahead):
        original = ['The', 'cat', 'is', '<i>', 'great', '</i>']
        derived  = ['The', 'cat', 'is', '<protected>', 'great', '</protected>']
        is_protected      = lambda t: t in ('<protected>', '</protected>')
        is_opening        = lambda t: t == '<protected>'
        -> ['The', 'cat', 'is', '<i>', '<protected>', 'great', '</protected>', '</i>']
    """
    n1 = len(original_tokens)
    n2 = len(derived_tokens)
    result = []
    idx1 = 0
    idx2 = 0

    # ------------------------------------------------------------------ helpers

    def tokens_equivalent(tok1: str, tok2: str) -> bool:
        if tok1 == tok2:
            return True
        if not tok1.strip() and not tok2.strip():
            return True
        return False

    def _is_tag(token: str) -> bool:
        """Default tag detector: any <…> token."""
        if is_tag_token_func is not None:
            return is_tag_token_func(token)
        return token.startswith('<') and token.endswith('>')

    def try_merge_original_to_match_derived(start_idx: int, target: str) -> int:
        """
        Try to merge consecutive original tokens to match the derived token.
        Returns the number of original tokens that combine to match target (0 = no match).
        """
        if start_idx >= n1:
            return 0
        accumulated = ""
        for i in range(start_idx, min(start_idx + 10, n1)):
            accumulated += original_tokens[i]
            if accumulated == target:
                return i - start_idx + 1
        return 0

    def handle_opening_protected_tag(t2: str) -> None:
        """
        Port of the opening-tag logic from merge_tokens_with_auto_labels.

        1. Scan forward in derived to find the first non-tag word after t2.
        2. Search forward in original to find that anchor word.
        3. Flush all original tokens up to (but not including) the anchor word.
        4. Emit the opening protected tag and advance idx2.
        """
        nonlocal idx1, idx2

        # Step 1 – find the anchor word in derived
        next_idx = idx2 + 1
        target_word = None
        while next_idx < n2:
            next_token = derived_tokens[next_idx]
            if not _is_tag(next_token):
                target_word = next_token
                break
            next_idx += 1

        if target_word is not None:
            # Step 2 – search for the anchor word in original
            search_idx = idx1
            found = False
            while search_idx < n1:
                if tokens_equivalent(original_tokens[search_idx], target_word):
                    # Step 3 – flush originals up to (not including) the anchor
                    while idx1 < search_idx:
                        result.append(original_tokens[idx1])
                        if log:
                            print(f"Pre-tag flush: '{original_tokens[idx1]}'")
                        idx1 += 1
                    # Step 4 – emit the opening tag
                    result.append(t2)
                    if log:
                        print(f"Opening protected (lookahead): '{t2}'")
                    idx2 += 1
                    found = True
                    break
                search_idx += 1

            if not found:
                # Anchor word not in original — emit tag as-is
                result.append(t2)
                if log:
                    print(f"Opening protected (no anchor found): '{t2}'")
                idx2 += 1
        else:
            # No non-tag word after the opening tag — emit as-is
            result.append(t2)
            if log:
                print(f"Opening protected (no target): '{t2}'")
            idx2 += 1

    # ---------------------------------------------------------------- main loop

    while idx1 < n1 and idx2 < n2:
        t1 = original_tokens[idx1]
        t2 = derived_tokens[idx2]

        if tokens_equivalent(t1, t2):
            result.append(t1)
            if log:
                print(f"Match: '{t1}' == '{t2}' -> '{t1}'")
            idx1 += 1
            idx2 += 1

        else:
            if is_protected_func(t2):
                # Protected token — distinguish opening from closing if possible
                if is_opening_protected_func is not None and is_opening_protected_func(t2):
                    handle_opening_protected_tag(t2)
                else:
                    # Closing tag (or undifferentiated protected token) — emit directly
                    result.append(t2)
                    if log:
                        print(f"Closing/atomic protected: '{t1}' vs '{t2}' -> '{t2}'")
                    idx2 += 1
            else:
                # Non-protected mismatch — try merging original tokens
                merge_count = try_merge_original_to_match_derived(idx1, t2)
                if merge_count > 0:
                    for i in range(merge_count):
                        result.append(original_tokens[idx1 + i])
                    if log:
                        merged = original_tokens[idx1:idx1 + merge_count]
                        print(f"Merged {merge_count} tokens: {merged} -> '{t2}'")
                    idx1 += merge_count
                    idx2 += 1
                else:
                    result.append(t1)
                    if log:
                        print(f"Diff: '{t1}' vs '{t2}' -> '{t1}'")
                    idx1 += 1

    # Append remaining original tokens
    if idx1 < n1:
        result.extend(original_tokens[idx1:])

    # Append remaining derived tokens (typically trailing protected tags)
    if idx2 < n2:
        result.extend(derived_tokens[idx2:])

    if log:
        print(f"   ✓ Merged {n1} original + {n2} derived → {len(result)} tokens")
        print(f"   ✓ Added {len(result) - n1} protected tokens")

    return result
