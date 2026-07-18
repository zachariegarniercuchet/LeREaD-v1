"""Sentence-based chunker — all spaCy logic lives here."""
from configs.config import MIN_TOKENS, CITATION_THRESHOLD
from src import (
    extract_body, tokenize, clean_tokens, decode,
    flatten_token_chunks, merge_tokens_general,
)
from src.output_control.protected_levenshtein_alignnment import apply_operations_safe, protected_levenshtein_distance


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
                           keep_manual_label=True, keep_auto_label=True,keep_bookmarks=True)

    doc                = nlp(decode(cleaned))
    raw_sentences      = [sent.text for sent in doc.sents]
    sentence_tokens    = [tokenize(s) for s in raw_sentences]
    flat_sentences     = flatten_token_chunks(sentence_tokens, separator="<sep>")

    is_sep = lambda t: t == "<sep>"
    #corrected = merge_tokens_general(
    #    original_tokens=cleaned,
    #    derived_tokens=flat_sentences,
    #    is_protected_func=is_sep,
    #    log=False,
    #)
    _, adjusted_operations = protected_levenshtein_distance(cleaned, flat_sentences, is_sep)
    corrected = apply_operations_safe(flat_sentences, adjusted_operations, is_sep)

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




def _debug_chunk_mismatch(cleaned, token_chunks):
    """
    Print useful information when chunk reconstruction fails.
    """
    reconstructed = flatten_token_chunks(token_chunks)

    print("=" * 80)
    print("CHUNK REASSEMBLY DEBUG")
    print("=" * 80)

    print(f"Original tokens      : {len(cleaned)}")
    print(f"Reconstructed tokens : {len(reconstructed)}")
    print()

    # ------------------------------------------------------------------
    # Find first mismatch
    # ------------------------------------------------------------------
    first_diff = None

    for i, (orig, recon) in enumerate(zip(cleaned, reconstructed)):
        if orig != recon:
            first_diff = i
            break

    if first_diff is None:
        if len(cleaned) != len(reconstructed):
            first_diff = min(len(cleaned), len(reconstructed))
            print("Sequences are identical until one ends.\n")
        else:
            print("No mismatch found!")
            return

    print(f"First mismatch at global token {first_diff}")
    print()

    # ------------------------------------------------------------------
    # Show local context
    # ------------------------------------------------------------------
    ctx = 15
    start = max(0, first_diff - ctx)
    end = min(max(len(cleaned), len(reconstructed)), first_diff + ctx)

    print("Original context:")
    print(cleaned[start:end])
    print()

    print("Reconstructed context:")
    print(reconstructed[start:end])
    print()

    # ------------------------------------------------------------------
    # Find which chunk contains the mismatch
    # ------------------------------------------------------------------
    offset = 0

    for chunk_idx, chunk in enumerate(token_chunks):

        if first_diff < offset + len(chunk):

            local = first_diff - offset

            expected = cleaned[offset:offset + len(chunk)]

            print("=" * 80)
            print(f"Mismatch is in chunk #{chunk_idx}")
            print(f"Chunk starts at global token {offset}")
            print(f"Chunk length: {len(chunk)}")
            print(f"Local index: {local}")
            print("=" * 80)
            print()

            print("Expected chunk:")
            print(expected)
            print()

            print("Actual chunk:")
            print(chunk)
            print()

            if local < len(expected) and local < len(chunk):
                print("Expected token :", repr(expected[local]))
                print("Actual token   :", repr(chunk[local]))

            break

        offset += len(chunk)

    print("=" * 80)