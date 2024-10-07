import numpy as np

from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def y(self) -> np.ndarray:
        """Returns the target vector of the dataset"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Returns length of the dataset"""
        pass
