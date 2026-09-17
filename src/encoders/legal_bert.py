from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .base import Encoder


class LegalBertEncoder(Encoder):
    """
    Domain-pretrained BERT (Legal-BERT and friends) used as a bi-encoder.

    A warning worth reading before you trust the numbers: nlpaueb/legal-bert is
    a masked-LM checkpoint, not a sentence-similarity model. Its raw pooled
    vectors are anisotropic and usually score *below* BGE-small on retrieval
    despite the domain advantage. Treat it as a domain baseline, and if it
    underperforms that is the expected result, not a bug in this class. The fix
    is to fine-tune it on (profile text, gold candidate metadata) pairs with
    MultipleNegativesRankingLoss, then point `model_name` at your checkpoint.

    Useful checkpoints:
        nlpaueb/legal-bert-base-uncased          (EU/US legal corpora)
        nlpaueb/legal-bert-small-uncased
        pile-of-law/legalbert-large-1.7M-2
        law-ai/InLegalBERT
    """

    MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

    similarity = "cosine"

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 256,
        model_name: str | None = None,
        pooling: str = "mean",
    ):
        if pooling not in {"mean", "cls"}:
            raise ValueError(f"Unsupported pooling: {pooling!r} (use 'mean' or 'cls')")

        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling = pooling
        self.model_name = model_name or self.MODEL_NAME

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @property
    def name(self) -> str:
        return "legal-bert"

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
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            hidden = self.model(**encoded).last_hidden_state

            if self.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.append(pooled.cpu().numpy())

        return np.concatenate(outputs, axis=0).astype(np.float32)