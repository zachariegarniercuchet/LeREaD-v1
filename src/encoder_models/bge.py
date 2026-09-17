from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import Encoder


class BGESmallEncoder(Encoder):
    """
    BGE-small-en-v1.5 encoder.

    Used for dense semantic representations of candidate metadata.
    """

    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = 64,
    ):
        self.batch_size = batch_size

        self.model = SentenceTransformer(
            self.MODEL_NAME,
            device=device,
        )

    @property
    def name(self) -> str:
        return "bge-s"

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.astype(np.float32)