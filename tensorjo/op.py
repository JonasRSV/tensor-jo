"""This module defines the structure of the op in the graph."""
from tensorjo import tensor
from abc import abstractmethod
import numpy as np


class Op():
    """An operation in the graph.

    The operation should implement a forward pass. The forward pass
    should act as a monoid for two tensors.

    The operation should implement two backward passes. The backward pass
    should enable gradient calculation with respect to any of the inputs.

    'backward_first' is the gradient wrt to the first
    argument to the forward pass

    'backward_second' is the gradient wrt to the second
    argument to the forward pass
    """

    @abstractmethod
    def forward(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Forward pass in the graph.

        Should always be a monoid for the tensors. No additional state.
        """
        pass

    @abstractmethod
    def backward_first(self, first: np.ndarray) -> np.ndarray:
        """Backward pass in the graph.

        Returns the gradients of first wrt output
        """
        pass

    @abstractmethod
    def backward_second(self, second: np.ndarray) -> np.ndarray:
        """Backward pass in the graph.

        Returns the gradients of second wrt output
        """
        pass
