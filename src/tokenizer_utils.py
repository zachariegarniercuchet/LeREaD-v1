import re

def tokenize(html_body: str, print=False) -> list:
    """
    Split HTML into a sequence of tokens that preserves:
    - HTML tags as single tokens (e.g., '<div class="x">')
    - Whitespace runs as separate tokens (spaces, newlines, tabs)
    - Punctuation as separate tokens (e.g., ',', '.', ';')
    - Words as separate tokens
    This ensures decode(tokens) == html_body with simple join AND prevents
    merging of words with punctuation after tag removal.
    """
    # Enhanced pattern: separates tags, whitespace, punctuation, words, and other chars
    # Group 1: HTML tags
    # Group 2: Whitespace runs
    # Group 3: Common punctuation (as separate tokens)
    # Group 4: Word characters (alphanumeric + underscore)
    # Group 5: Any other single character
    pattern = re.compile(r"(<[^>]*>)|(\s+)|([.,;:!?()[\]{}\"'`‑–—])|(\w+)|([^\w\s<>])")
    tokens = []
    for m in pattern.finditer(html_body):
        tok = m.group(1) or m.group(2) or m.group(3) or m.group(4) or m.group(5)
        if tok:  # Safety check
            tokens.append(tok)

    if print:
        print(f"   ✓ Tokenized into {len(tokens)} tokens (tags+whitespace+punctuation+words, reversible)")
    return tokens


def decode(tokens: list, print=False) -> str:
    """
    Reconstruct HTML body by concatenating tokens exactly.
    """
    reconstructed_html = "".join(tokens)

    if print:
        print(f"   ✓ Decoded {len(tokens)} tokens into HTML")
    return reconstructed_html