"""This module defines the structure of the op in the graph."""
import typecheck as tc
from tensorjo import tensor
from abc import abstractmethod


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
    @tc.typecheck
    def forward(self, m1: tensor, m2: tensor):
        """Forward pass in the graph.

        Should always be a monoid for the tensors. No additional state.
        """
        pass

    @abstractmethod
    @tc.typecheck
    def backward_first(self, first: tensor, output: tensor):
        """Backward pass in the graph.

        Returns the gradients of wrt with respect to output
        """
        pass

    @abstractmethod
    @tc.typecheck
    def backward_second(self, second: tensor, output: tensor):
        """Backward pass in the graph.

        Returns the gradients of wrt with respect to output
        """
        pass
