"""Sentence-based chunker — all spaCy logic lives here."""
from config import MIN_TOKENS, CITATION_THRESHOLD
from src import (
    extract_body, tokenize, clean_tokens, decode,
    flatten_token_chunks, merge_tokens_general,
)


# ── Citation detection ────────────────────────────────────────────────────────

def _combined_density(sentence: str) -> float:
    if len(sentence) < 10:
        return 0.0
    period_density = (sentence.count(".") / len(sentence)) * 100
    digit_density  = (sum(c.isdigit() for c in sentence) / len(sentence)) * 100
    return period_density + digit_density


def _detect_citation_sections(sentences: list[str], threshold: float, consecutive_gap: int = 3) -> list[bool]:
    is_citation            = [False] * len(sentences)
    in_section             = False
    section_start          = None
    below_count            = 0
    section_ended          = False

    for i, sent in enumerate(sentences):
        if section_ended:
            break

        if _combined_density(sent) > threshold:
            if not in_section:
                in_section    = True
                section_start = i
            below_count = 0
            is_citation[i] = True
        else:
            if in_section:
                below_count += 1
                if below_count >= consecutive_gap:
                    for j in range(section_start, i - consecutive_gap + 1):
                        is_citation[j] = True
                    section_ended = True
                    in_section    = False

    if in_section and section_start is not None:
        end = len(sentences) - 1 - below_count
        for j in range(section_start, end + 1):
            is_citation[j] = True

    return is_citation


# ── Sentence merging ──────────────────────────────────────────────────────────

def _merge_sentences(tokens: list[str], citation_threshold: float, min_tokens: int) -> list[str]:
    """Flat token list in (with <sep>) → flat token list out (with selective <sep>)."""
    if not tokens:
        return []

    # Split on <sep>
    sentences_tokens: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        if tok == "<sep>":
            if current:
                sentences_tokens.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        sentences_tokens.append(current)

    sentences_text = [decode(s) for s in sentences_tokens]
    is_citation    = _detect_citation_sections(sentences_text, threshold=citation_threshold)

    result: list[str] = []
    chunk_size = 0

    for i, sent_tokens in enumerate(sentences_tokens):
        result.extend(sent_tokens)
        chunk_size += len(sent_tokens)

        if i < len(sentences_tokens) - 1:
            last = sent_tokens[-1] if sent_tokens else ""
            keep_sep = (
                chunk_size >= min_tokens
                and (last == ";" if is_citation[i] else last == ".")
            )
            if keep_sep:
                result.append("<sep>")
                chunk_size = 0

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def compute_sentence_chunks(
    html_content: str,
    nlp,
    min_tokens: int          = MIN_TOKENS,
    citation_threshold: float = CITATION_THRESHOLD,
) -> list[list[str]]:
    """Full pipeline: HTML → list of token chunks."""
    body    = extract_body(html_content)
    tokens  = tokenize(body)
    cleaned = clean_tokens(html_tokens=tokens, normalize=True,
                           keep_manual_label=True, keep_bookmarks=True)

    doc                = nlp(decode(cleaned))
    raw_sentences      = [sent.text for sent in doc.sents]
    sentence_tokens    = [tokenize(s) for s in raw_sentences]
    flat_sentences     = flatten_token_chunks(sentence_tokens, separator="<sep>")

    is_sep = lambda t: t == "<sep>"
    corrected = merge_tokens_general(
        original_tokens=cleaned,
        derived_tokens=flat_sentences,
        is_protected_func=is_sep,
        log=False,
    )

    flat_chunks = _merge_sentences(corrected, citation_threshold, min_tokens)

    # Split on <sep> → final chunk list
    token_chunks: list[list[str]] = []
    current: list[str] = []
    for tok in flat_chunks:
        if tok == "<sep>":
            token_chunks.append(current)
            current = []
        else:
            current.append(tok)
    token_chunks.append(current)

    assert flatten_token_chunks(token_chunks) == cleaned, \
        "Chunk reassembly mismatch — check chunking logic."

    return token_chunks