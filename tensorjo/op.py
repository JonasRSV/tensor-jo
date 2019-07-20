"""This module defines the structure of the op in the graph."""
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
    def forward(self, *args) -> np.ndarray:
        """Forward pass in the graph.

        Should always be a monoid for the tensors. No additional state.
        """
        pass

    def backward_functor(self) -> np.ndarray:
        """Backward pass in the graph.

        Returns the gradient wrt to the functor.
        """
        raise NotImplementedError("Backward_functor is not implemented.")

    def backward_first(self) -> np.ndarray:
        """Backward pass in the graph.

        Returns the gradients of first wrt output
        """
        raise NotImplementedError("Backward_first is not implemented.")

    def backward_second(self) -> np.ndarray:
        """Backward pass in the graph.

        Returns the gradients of second wrt output
        """
        raise NotImplementedError("Backward_second is not implemented.")

    @abstractmethod
    def cache(self) -> np.ndarray:
        """Return output from previous forward pass."""
        pass

    @abstractmethod
    def shape(self) -> tuple:
        """Return the output shape of this op."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Name of the op."""
        pass
