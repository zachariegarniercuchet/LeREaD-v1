from typing import List

import numpy as np

from .base import Encoder


class Qwen3EmbeddingEncoder(Encoder):
    """
    Qwen3-Embedding (LLM-based dense encoder), loaded through SentenceTransformers.

    Sizes: Qwen/Qwen3-Embedding-0.6B | -4B | -8B.
    The 8B variant needs roughly 16 GB of VRAM in fp16/bf16, so keep the batch
    size small and pass --device cuda. Start with 0.6B to validate the plumbing,
    then swap the model name once you are ready to pay for the big one.

    Qwen3-Embedding is instruction-tuned: queries get a task instruction, the
    documents do not. The instruction genuinely moves the numbers, so it is
    spelled out here for the legal-citation task rather than left generic.
    """

    MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

    DEFAULT_INSTRUCTION = (
        "Given a legal citation or case reference, retrieve the metadata record "
        "of the matching CanLII document."
    )

    similarity = "cosine"

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = 8,
        model_name: str | None = None,
        max_length: int = 512,
        instruction: str | None = None,
        torch_dtype: str = "auto",
    ):
        from sentence_transformers import SentenceTransformer

        self.batch_size = batch_size
        self.model_name = model_name or self.MODEL_NAME
        self.instruction = instruction or self.DEFAULT_INSTRUCTION

        model_kwargs = {}
        if torch_dtype != "auto":
            import torch

            model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)

        self.model = SentenceTransformer(
            self.model_name,
            device=device,
            model_kwargs=model_kwargs or None,
            tokenizer_kwargs={"padding_side": "left"},
        )
        self.model.max_seq_length = max_length

    @property
    def name(self) -> str:
        return "qwen3-emb"

    def _encode(self, texts: List[str], prompt: str | None = None) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            prompt=prompt,
        )
        return embeddings.astype(np.float32)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        prompt = f"Instruct: {self.instruction}\nQuery: "
        return self._encode(texts, prompt=prompt)