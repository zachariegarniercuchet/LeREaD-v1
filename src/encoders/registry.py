"""
Encoder registry.

Imports are lazy on purpose: loading this module should not drag in
sentence-transformers or a multi-gigabyte Qwen checkpoint when the user only
asked for BM25. Each factory imports its own dependency the first time it is
called.

Adding a new dense/sparse model is a two-line change: write the Encoder
subclass, then register a factory below.
"""

from typing import Callable, Dict

from .base import Encoder


def _bge_small(**kwargs) -> Encoder:
    from .bge import BGESmallEncoder
    return BGESmallEncoder(**kwargs)

def _bge_large(**kwargs) -> Encoder:
    from .bge import BGELargeEncoder

    kwargs.setdefault(
        "model_name",
        "/home/z/zagar/links/scratch/bge-large-en-v1.5",
    )
    return BGELargeEncoder(**kwargs)


def _splade(**kwargs) -> Encoder:
    from .splade import SPLADEEncoder
    return SPLADEEncoder(**kwargs)


def _legal_bert(**kwargs) -> Encoder:
    from .legal_bert import LegalBertEncoder

    kwargs.setdefault(
        "model_name",
        "/home/z/zagar/links/scratch/legal-bert-base-uncased",
    )
    return LegalBertEncoder(**kwargs)


def _qwen3_0_6b(**kwargs) -> Encoder:
    from .qwen import Qwen3EmbeddingEncoder
    kwargs.setdefault("model_name", "Qwen/Qwen3-Embedding-0.6B")
    return Qwen3EmbeddingEncoder(**kwargs)


def _qwen3_8b(**kwargs) -> Encoder:
    from .qwen import Qwen3EmbeddingEncoder
    kwargs.setdefault("model_name", "/home/z/zagar/links/scratch/Qwen3-Embedding-8B")
    kwargs.setdefault("batch_size", 4)
    return Qwen3EmbeddingEncoder(**kwargs)


ENCODER_REGISTRY: Dict[str, Callable[..., Encoder]] = {
    "bge-s": _bge_small,
    "bge-l":_bge_large,
    "splade": _splade,
    "legal-bert": _legal_bert,
    "qwen3-0.6b": _qwen3_0_6b,
    "qwen3-8b": _qwen3_8b,
}


def create_encoder(encoder_name: str, **kwargs) -> Encoder:
    """Create an encoder from its registered name."""

    if encoder_name not in ENCODER_REGISTRY:
        available = ", ".join(sorted(ENCODER_REGISTRY))
        raise ValueError(
            f"Unknown encoder '{encoder_name}'. Available encoders: {available}"
        )

    return ENCODER_REGISTRY[encoder_name](**kwargs)