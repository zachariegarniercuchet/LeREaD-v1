"""
Retriever registry — the single place the CLI's `--retriever` choices come from.

To add a model:
  * dense/sparse vector model -> register the Encoder in
    src/encoder_models/registry.py, then add one line here pointing at
    VectorRetriever.
  * anything that is not a vector (lexical, API-based reranker, hybrid) ->
    write a Retriever subclass and register it here.

Nothing else in the codebase needs to change.
"""

from typing import Callable, Dict, List

from .base import Retriever
from .bm25 import BM25Retriever
from .dense import VectorRetriever


def _vector(encoder_name: str) -> Callable[..., Retriever]:
    def factory(**kwargs) -> Retriever:
        return VectorRetriever(encoder_name=encoder_name, **kwargs)
    return factory


RETRIEVER_REGISTRY: Dict[str, Callable[..., Retriever]] = {
    "bm25": lambda **kwargs: BM25Retriever(**kwargs),
    "bge-s": _vector("bge-s"),
    "splade": _vector("splade"),
    "legal-bert": _vector("legal-bert"),
    "qwen3-0.6b": _vector("qwen3-0.6b"),
    "qwen3-8b": _vector("qwen3-8b"),
}


def available_retrievers() -> List[str]:
    return sorted(RETRIEVER_REGISTRY)


def create_retriever(retriever_name: str, **kwargs) -> Retriever:
    if retriever_name not in RETRIEVER_REGISTRY:
        available = ", ".join(available_retrievers())
        raise ValueError(
            f"Unknown retriever '{retriever_name}'. Available: {available}"
        )
    return RETRIEVER_REGISTRY[retriever_name](**kwargs)