"""
BM25 lexical retriever.

Two things to keep in mind when you compare it against the dense models:

1. BM25 scores are unbounded and depend on query length and corpus statistics.
   A score of 18.4 for one profile and 3.1 for another says nothing about which
   match is more reliable. `--score-threshold` is therefore meaningless here,
   and main_resolution refuses it rather than silently producing garbage.

2. Tokenization is the whole ballgame for legal citations. `"[1986] 1 SCR 863"`
   split on whitespace gives `["[1986]", "1", "scr", "863"]` — the brackets stay
   glued to the year, so a query writing `1986` never matches the document's
   `[1986]`. The regex tokenizer below strips punctuation, which on citation-
   heavy text is usually worth several points of recall over `str.split()`.
"""

import json
import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .base import Retriever, _topk_from_scores

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class BM25Retriever(Retriever):

    name = "bm25"
    score_kind = "score"

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: str = "regex",
        **_ignored,
    ):
        # **_ignored swallows device/batch_size so the CLI can pass the same
        # kwargs to every retriever without special-casing.
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.bm25: Optional[BM25Okapi] = None
        self._size = 0

    # -- tokenization --------------------------------------------------------

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        if self.tokenizer == "split":
            return text.split()
        return _TOKEN_RE.findall(text)

    # -- indexing ------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    def build(self, texts: List[str]) -> None:
        corpus = [self.tokenize(t) for t in texts]

        # rank_bm25 divides by the average document length, which is NaN on an
        # all-empty corpus; fail loudly instead.
        if not any(corpus):
            raise ValueError("Every candidate tokenized to an empty document.")

        self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        self._size = len(corpus)

    def save(self, out_dir: Path) -> None:
        if self.bm25 is None:
            raise RuntimeError("Nothing to save: call build() first.")

        out_dir.mkdir(parents=True, exist_ok=True)

        with (out_dir / "bm25.pkl").open("wb") as f:
            pickle.dump({"bm25": self.bm25, "size": self._size}, f)

        with (out_dir / "retriever.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "type": "bm25",
                    "k1": self.k1,
                    "b": self.b,
                    "tokenizer": self.tokenizer,
                    "size": self._size,
                },
                f,
                indent=2,
            )

    def load(self, out_dir: Path) -> None:
        index_path = out_dir / "bm25.pkl"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing BM25 index: {index_path}")

        with index_path.open("rb") as f:
            payload = pickle.load(f)

        # tolerate an index pickled as a bare BM25Okapi by an older run
        if isinstance(payload, dict):
            self.bm25 = payload["bm25"]
            self._size = payload.get("size", len(self.bm25.doc_freqs))
        else:
            self.bm25 = payload
            self._size = len(self.bm25.doc_freqs)

        meta_path = out_dir / "retriever.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            self.tokenizer = meta.get("tokenizer", self.tokenizer)

    # -- search --------------------------------------------------------------

    def search(
        self,
        queries: List[str],
        k: int = 1,
        subset_idx: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        if self.bm25 is None:
            raise RuntimeError("Index not loaded: call build() or load() first.")
        if not queries:
            return np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=np.float32)

        # BM25Okapi always scores the entire corpus; the subset is applied
        # afterwards. Note that the IDF and average-length statistics stay
        # those of the full pool, which is the correct behaviour — the doctype
        # filter is a post-hoc restriction, not a separate corpus.
        rows = []
        for query in queries:
            scores = self.bm25.get_scores(self.tokenize(query))
            rows.append(np.asarray(scores, dtype=np.float32))

        all_scores = np.vstack(rows)

        if subset_idx is not None:
            subset_idx = np.asarray(subset_idx)
            if subset_idx.size == 0:
                raise ValueError("Empty candidate subset passed to search().")
            all_scores = all_scores[:, subset_idx]

        return _topk_from_scores(all_scores, subset_idx, k)

    def describe(self) -> dict:
        info = super().describe()
        info.update({"k1": self.k1, "b": self.b, "tokenizer": self.tokenizer})
        return info