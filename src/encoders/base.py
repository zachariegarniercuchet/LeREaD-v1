from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Encoder(ABC):
    """
    Base interface for *dense/sparse vector* encoders.

    An encoder maps a list of strings to one vector per string, shape:

        (number_of_texts, representation_dimension)

    Important: BM25 is NOT an encoder. It has no per-document vector and no
    meaningful `encode()` output. It lives in src/retrievers/bm25.py instead.

    Query / document asymmetry
    --------------------------
    Several modern retrieval models expect queries and documents to be encoded
    differently (BGE prepends an instruction to queries, Qwen3-Embedding uses a
    task prompt, etc.). The retriever always calls `encode_documents()` at index
    time and `encode_queries()` at search time, so a model that needs the
    asymmetry can implement it and a model that doesn't simply inherits the
    default (both delegate to `encode()`).
    """

    #: "cosine" for normalized dense models, "dot" for SPLADE-style sparse ones.
    similarity: str = "cosine"

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name used for cache directories."""
        raise NotImplementedError

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into a 2D float32 array."""
        raise NotImplementedError

    # -- overridable hooks ---------------------------------------------------

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        return self.encode(texts)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        return self.encode(texts)