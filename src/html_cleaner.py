def _strip_tags(tokens: list, keep_manual_label=False, keep_auto_label=False, keep_bookmarks=False, merge = True, log=False) -> list:
    """
    Remove tag tokens from the HTML token sequence and return text token list
    while preserving whitespace tokens exactly.
    - Drops any token that starts with '<' and ends with '>' (except manual_label tags if keep_manual_label=True)
    - Keeps whitespace and text tokens unchanged
    - If keep_manual_label=True: keeps all <manual_label ...> and </manual_label> tags
    """

    # --- REMOVE TAG TOKENS ----
    text_parts = []
    for t in tokens:
        if len(t) >= 2 and t[0] == '<' and t[-1] == '>':
            # Check if we should keep manual_label tags
            if keep_manual_label:
                # Keep manual_label tags (opening and closing)
                if t.lower().startswith('<manual_label') or t.lower().startswith('</manual_label'):
                    text_parts.append(t)
                    continue

            if keep_auto_label:
                # Keep auto_label tags (opening and closing)
                if t.lower().startswith('<auto_label') or t.lower().startswith('</auto_label'):
                    text_parts.append(t)
                    continue

            if keep_bookmarks:
                # Keep bookmarks tags (opening and closing)
                if t.lower().startswith('<htmllabelizer_bookmark') or t.lower().startswith('</htmllabelizer_bookmark>'):
                    text_parts.append(t)
                    continue
            # Skip all other tags
            continue
        text_parts.append(t)

    # --- MERGE WHITESPACE TOKENS ---- 
    if merge:
        merge_tokens = []
        accumulated_whitespace = ""
        
        for tok in text_parts:
            # Check if token is pure whitespace
            if tok and not tok.strip():
                # Accumulate whitespace
                accumulated_whitespace += tok
            else:
                # Non-whitespace token found
                # First, flush any accumulated whitespace
                if accumulated_whitespace:
                    merge_tokens.append(accumulated_whitespace)
                    accumulated_whitespace = ""
                # Then add the current non-whitespace token
                merge_tokens.append(tok)
        
        # Don't forget any trailing whitespace
        if accumulated_whitespace:
            merge_tokens.append(accumulated_whitespace)

    if log:
        print(f"   ✓ Stripped tags: {len(tokens)} -> text token length {len(text_parts)}")
    return text_parts if not merge else merge_tokens


def _normalize_text_tokens(text_tokens: list, log=False) -> list:
    """
    Normalize complex whitespace tokens by replacing them with a single space.
    
    Rules:
    - Tokens with text content (letters, numbers, punctuation): KEEP AS-IS
    - Simple whitespace tokens (' ', '\n', '\t'): KEEP AS-IS
    - Complex whitespace tokens ('\n\n\n\xa0\xa0\n\n\xa0\n'): REPLACE with ' '
    
    A complex whitespace token = pure whitespace with length > 1
    """
    normalized = []
    
    for tok in text_tokens:
        # If token has any non-whitespace character, keep as-is
        if tok.strip():
            normalized.append(tok)
        # Token is pure whitespace
        elif len(tok) == 1:
            # Simple single-character whitespace: keep as-is
            normalized.append(tok)
        else:
            # Complex multi-character whitespace: replace with single space
            normalized.append(' ')

    if log:
        print(f"   ✓ Normalized text tokens: {len(text_tokens)} -> {len(normalized)}")

    return normalized


def clean_tokens(html_tokens: list, normalize: bool = False, keep_manual_label=False, keep_auto_label=False, keep_bookmarks=False, log=False) -> list:
    """
    Produce a cleaned token list by:
    1) Removing all tag tokens and building plain text token list
    2) Optionally normalizing whitespace tokens (NBSPs and excessive newlines/spaces)
    Returns: cleaned tokens (no tag tokens)
    """
    text_tokens = _strip_tags(tokens=html_tokens, keep_manual_label=keep_manual_label, keep_auto_label=keep_auto_label, keep_bookmarks=keep_bookmarks, log=log)
    if normalize:
        text_tokens = _normalize_text_tokens(text_tokens, log=log)

    if log:
        print(f"   ✓ Cleaned tokens count: {len(text_tokens)} (normalize={normalize})")
    return text_tokens