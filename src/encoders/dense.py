"""
Vector retriever: wraps any `Encoder` (BGE, Qwen3, Legal-BERT, SPLADE).

One instance covers both the dense-cosine and the SPLADE-dot cases; the
encoder declares which via `Encoder.similarity`. That is the only difference
between them, so there is no reason for two classes.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.encoders import Encoder, create_encoder

from .base import Retriever, _topk_from_scores


class VectorRetriever(Retriever):

    score_kind = "similarity"

    def __init__(
        self,
        encoder_name: str,
        encoder: Optional[Encoder] = None,
        **encoder_kwargs,
    ):
        self.encoder_name = encoder_name
        self._encoder = encoder
        self._encoder_kwargs = encoder_kwargs
        self.matrix: Optional[np.ndarray] = None

    # -- lazy encoder --------------------------------------------------------
    # At index time the encoder is always needed. At search time it is also
    # needed (to embed queries), but loading it should not happen just because
    # someone imported the module.

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = create_encoder(self.encoder_name, **self._encoder_kwargs)
        return self._encoder

    @property
    def name(self) -> str:
        return self.encoder_name

    @property
    def size(self) -> int:
        return 0 if self.matrix is None else int(self.matrix.shape[0])

    # -- indexing ------------------------------------------------------------

    def build(self, texts: List[str]) -> None:
        matrix = self.encoder.encode_documents(texts)
        self.matrix = np.asarray(matrix, dtype=np.float32)

    def save(self, out_dir: Path) -> None:
        if self.matrix is None:
            raise RuntimeError("Nothing to save: call build() first.")

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "embeddings.npy", self.matrix)

        with (out_dir / "retriever.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "type": "vector",
                    "encoder": self.encoder_name,
                    "similarity": self.encoder.similarity,
                    "shape": list(self.matrix.shape),
                },
                f,
                indent=2,
            )

    def load(self, out_dir: Path) -> None:
        embeddings_path = out_dir / "embeddings.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")
        self.matrix = np.load(embeddings_path)

    # -- search --------------------------------------------------------------

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / (norms + 1e-12)

    def search(
        self,
        queries: List[str],
        k: int = 1,
        subset_idx: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        if self.matrix is None:
            raise RuntimeError("Index not loaded: call build() or load() first.")
        if not queries:
            return np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=np.float32)

        doc_matrix = self.matrix if subset_idx is None else self.matrix[subset_idx]
        if doc_matrix.shape[0] == 0:
            raise ValueError("Empty candidate subset passed to search().")

        query_matrix = np.asarray(
            self.encoder.encode_queries(queries), dtype=np.float32
        )

        if self.encoder.similarity == "cosine":
            query_matrix = self._l2_normalize(query_matrix)
            doc_matrix = self._l2_normalize(doc_matrix)

        scores = query_matrix @ doc_matrix.T

        return _topk_from_scores(scores, subset_idx, k)

    def describe(self) -> dict:
        info = super().describe()
        info["similarity"] = self.encoder.similarity
        info["encoder"] = self.encoder_name
        if self.matrix is not None:
            info["embedding_shape"] = list(self.matrix.shape)
        return info