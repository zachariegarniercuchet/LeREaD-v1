"""Paragraph-based chunker — BeautifulSoup logic lives here."""
from bs4 import BeautifulSoup
from config import MIN_TOKENS
from src import extract_body, tokenize, clean_tokens


def compute_paragraph_chunks(
    html_content: str,
    min_tokens: int = MIN_TOKENS,
) -> list[list[str]]:
    """Full pipeline: HTML → list of token chunks."""
    body = extract_body(html_content)
    soup = BeautifulSoup(body, "html.parser")

    # Leaf block elements only (no nested block children)
    leaf_blocks = [
        child
        for child in soup.find_all(
            ["p", "li", "blockquote", "pre", "h1", "h2", "h3", "h4", "h5", "h6"]
        )
        if not child.find(["p", "li", "blockquote", "pre"])
    ]

    paragraph_token_lists: list[list[str]] = []
    for block in leaf_blocks:
        cleaned = clean_tokens(
            html_tokens=tokenize(str(block)),
            normalize=True,
            keep_manual_label=True,
            keep_bookmarks=True,
        )
        if cleaned:
            paragraph_token_lists.append(cleaned)

    # Merge until min_tokens
    token_chunks: list[list[str]] = []
    current: list[str] = []
    for para in paragraph_token_lists:
        if not current:
            current = list(para)
        elif len(current) >= min_tokens:
            token_chunks.append(current)
            current = list(para)
        else:
            current.extend(para)
    if current:
        token_chunks.append(current)

    return token_chunks