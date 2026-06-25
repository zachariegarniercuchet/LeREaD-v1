"""
Single entry point for all chunking.

Usage (lazy, one file at a time):
    chunks = ChunkerFactory.get_chunks(
        html_content, method="sentence",
        split="train", filename="doc_001",
    )
"""
from __future__ import annotations
from configs.config import CHUNK_CACHE_DIR, MIN_TOKENS
from .cache import cache_exists, load_cache, save_cache


class ChunkerFactory:

    @staticmethod
    def get_chunks(
        html_content: str,
        method: str        = "sentence",
        annotated=False,  # if True, use the annotated cache dir; else use the unannotated cache dir
        min_tokens: int    = MIN_TOKENS,
        split: str | None  = None,
        filename: str | None = None,
        nlp                = None,       # required for method="sentence" on cache miss
    ) -> list[list[str]]:
        """
        Return token chunks for one document.

        - Cache hit  → return immediately (spaCy never loaded).
        - Cache miss → compute, persist, return.

        `split` + `filename` are both required for caching.
        If either is None the result is computed but not cached.
        """

        if annotated:
            cache_dir = f"{CHUNK_CACHE_DIR}/annotated"
        else:
            cache_dir = f"{CHUNK_CACHE_DIR}/original"
        use_cache = split is not None and filename is not None

        if use_cache and cache_exists(cache_dir=cache_dir, method=method, split=split, filename=filename):
            print(f"✓ cache hit  [{method}] {split}/{filename}")
            return load_cache(cache_dir=cache_dir, method=method, split=split, filename=filename)

        chunks = ChunkerFactory._compute(html_content, method, min_tokens, nlp)

        if use_cache:
            save_cache(cache_dir=cache_dir, method=method, split=split, filename=filename, token_chunks=chunks)
            print(f"✓ cache saved [{method}] {split}/{filename}")

        return chunks

    # ── private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _compute(
        html_content: str,
        method: str,
        min_tokens: int,
        nlp,
    ) -> list[list[str]]:
        if method == "sentence":
            if nlp is None:
                raise ValueError(
                    "method='sentence' requires an `nlp` model on a cache miss. "
                    "Pass nlp=spacy.load('en_core_web_trf') or precompute the cache."
                )
            from .sentence import compute_sentence_chunks
            return compute_sentence_chunks(html_content, nlp, min_tokens=min_tokens)

        elif method == "paragraph":
            from .paragraph import compute_paragraph_chunks
            return compute_paragraph_chunks(html_content, min_tokens=min_tokens)

        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'sentence' or 'paragraph'.")