from abc import ABC, abstractmethod
from typing import List
import numpy as np


class Encoder(ABC):
    """
    Base interface for metadata encoders.

    An encoder takes a list of text strings and returns one vector per text.
    """

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into a 2D numpy array of shape:

            (number_of_texts, embedding_dimension)
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name used for cache directories."""
        raise NotImplementedError