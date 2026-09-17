"""
Retriever interface.

Why this layer exists
---------------------
`Encoder` is the wrong abstraction for BM25. BM25 has no per-document vector,
no fixed dimension, and no similarity function you can apply to a pair of rows:
its score is a function of (query tokens, document, whole-corpus statistics).
Forcing it into `encode() -> matrix` is what made the precompute script
special-case it and made main_resolution's `_top_k()` cosine unusable.

`Retriever` is the abstraction that all three families actually share:

    build(texts)                     index the candidate pool
    save(dir) / load(dir)            persist that index
    search(queries, k, subset_idx)   return (global_indices, scores)

Everything downstream (main_resolution) talks only to this interface, so adding
a model never touches the resolution logic again.

`subset_idx` is how doctype filtering is expressed. It is a numpy array of
*global* candidate row indices to restrict the search to, or None for the full
pool. Returned indices are always global, so the caller can index
`candidates[i]` without any bookkeeping.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class Retriever(ABC):

    #: identifier written into config.json and used for cache dir names
    name: str = "retriever"

    #: "similarity" (higher is better, bounded, comparable across queries) or
    #: "score" (higher is better, unbounded, NOT comparable across queries).
    #: Thresholding only makes sense for the former — see main_resolution.
    score_kind: str = "similarity"

    @abstractmethod
    def build(self, texts: List[str]) -> None:
        """Index the candidate pool."""
        raise NotImplementedError

    @abstractmethod
    def save(self, out_dir: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, out_dir: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        queries: List[str],
        k: int = 1,
        subset_idx: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            indices: (n_queries, k) array of GLOBAL candidate indices
            scores:  (n_queries, k) array of scores, descending per row
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of indexed candidates (used to sanity-check candidates.json)."""
        raise NotImplementedError

    def describe(self) -> dict:
        """Metadata dumped into config.json."""
        return {"name": self.name, "score_kind": self.score_kind, "size": self.size}


def _topk_from_scores(
    scores: np.ndarray,
    subset_idx: Optional[np.ndarray],
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shared helper: given a (n_queries, n_subset) score matrix, return the top-k
    global indices and their scores.

    Uses argpartition then sorts only the k survivors — argsort over the whole
    pool per query is wasteful once the pool grows past a few thousand rows.
    """

    n_queries, n_subset = scores.shape
    k = max(1, min(k, n_subset))

    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)

    order = np.argsort(-part_scores, axis=1)
    local_idx = np.take_along_axis(part, order, axis=1)
    top_scores = np.take_along_axis(part_scores, order, axis=1)

    if subset_idx is None:
        global_idx = local_idx
    else:
        global_idx = np.asarray(subset_idx)[local_idx]

    return global_idx, top_scores