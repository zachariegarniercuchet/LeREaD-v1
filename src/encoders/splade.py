from typing import List

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .base import Encoder


class SPLADEEncoder(Encoder):
    """
    SPLADE sparse encoder.

    Produces a sparse vocabulary-sized representation for each text.

    The current implementation stores the result as a dense float32 numpy
    matrix for simplicity. If the candidate pool becomes very large, this
    should be changed to scipy.sparse CSR storage.

    Note the `similarity = "dot"`: SPLADE scores are dot products, and the term
    weights carry magnitude information that cosine normalization destroys.
    """

    MODEL_NAME = "naver/splade-cocondenser-ensembledistil"

    similarity = "dot"

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 256,
        model_name: str | None = None,
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_name = model_name or self.MODEL_NAME

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)

        self.model.to(self.device)
        self.model.eval()

    @property
    def name(self) -> str:
        return "splade"

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        outputs = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            logits = self.model(**encoded).logits

            # SPLADE pooling: log(1 + ReLU(logits)), max-pooled over the sequence.
            activations = torch.log1p(torch.relu(logits))

            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            activations = activations.masked_fill(attention_mask == 0, 0)

            sparse_representation = torch.max(activations, dim=1).values

            outputs.append(sparse_representation.cpu().numpy())

        return np.concatenate(outputs, axis=0).astype(np.float32)