from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import Encoder


class BGESmallEncoder(Encoder):
    """
    bge-large-en-v1.5 dense encoder.

    BGE is trained asymmetrically: queries are expected to carry a short
    instruction prefix, passages are encoded bare. Skipping the prefix costs a
    couple of points of recall, so it is applied in `encode_queries()`.
    """

    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    similarity = "cosine"

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = 64,
        model_name: str | None = None,
        use_query_instruction: bool = True,
    ):
        self.batch_size = batch_size
        self.use_query_instruction = use_query_instruction
        self.model_name = model_name or self.MODEL_NAME

        self.model = SentenceTransformer(self.model_name, device=device)

    @property
    def name(self) -> str:
        return "bge-s"

    def _encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        if self.use_query_instruction:
            texts = [self.QUERY_INSTRUCTION + t for t in texts]
        return self._encode(texts)

class BGELargeEncoder(Encoder):
    """
    BGE-small-en-v1.5 dense encoder.

    BGE is trained asymmetrically: queries are expected to carry a short
    instruction prefix, passages are encoded bare. Skipping the prefix costs a
    couple of points of recall, so it is applied in `encode_queries()`.
    """

    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    similarity = "cosine"

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = 64,
        model_name: str | None = None,
        use_query_instruction: bool = True,
    ):
        self.batch_size = batch_size
        self.use_query_instruction = use_query_instruction
        self.model_name = model_name or self.MODEL_NAME

        self.model = SentenceTransformer(self.model_name, device=device)

    @property
    def name(self) -> str:
        return "bge-s"

    def _encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        if self.use_query_instruction:
            texts = [self.QUERY_INSTRUCTION + t for t in texts]
        return self._encode(texts)