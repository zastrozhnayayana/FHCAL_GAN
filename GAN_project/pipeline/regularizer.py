from abc import abstractmethod

import torch


class Regularizer:
    """
    Regularization loss
    """
    def step(self) -> None:
        """
        Call after each epoch
        """
        pass

    @abstractmethod
    def __call__(self) -> torch.Tensor:
        pass

__all__ = ['Regularizer']
