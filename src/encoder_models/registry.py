from typing import Callable

from .base import Encoder
from .bge import BGESmallEncoder
from .splade import SPLADEEncoder


ENCODER_REGISTRY: dict[str, Callable[..., Encoder]] = {
    "bge-s": BGESmallEncoder,
    "splade": SPLADEEncoder,
}


def create_encoder(
    encoder_name: str,
    **kwargs,
) -> Encoder:
    """
    Create an encoder from its registered name.
    """

    if encoder_name not in ENCODER_REGISTRY:
        available = ", ".join(sorted(ENCODER_REGISTRY))

        raise ValueError(
            f"Unknown encoder '{encoder_name}'. "
            f"Available encoders: {available}"
        )

    return ENCODER_REGISTRY[encoder_name](**kwargs)