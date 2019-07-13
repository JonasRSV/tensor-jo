"""This files defines the normal multiplication op."""
from tensorjo import op
from tensorjo import tensor
import typecheck as tc
import numpy as np


class multiplication(op.Op):
    """Implements the forward and backward pass for multiplication."""

    @tc.typecheck
    def __init__(self, m1: tensor, m2: tensor):
        """Initialize op.

        1. Make sure the two tensors can have this op applied to them.
        2. Need to store reference to the tensors for the gradient calculations

        Why store tensor? Why have stateful op? Because we are in many cases
        going to need the m2 tensor to calculate the gradient for the m1 tensor
        and vice-versa. The alternative is for the nodes to pass this
        information to the op. But that is a major hassle because then the
        nodes need to know what other nodes are used together with it
        in an op.
        """
        try:
            m1.v * m2.v
        except ValueError as e:
            raise ValueError(
                "Failed to construct multiplication op with tensors %s and %s "
                % (m1, m2) + "- %s" % e)

        self.m1 = m1
        self.m2 = m2

    def forward(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Implement the forward pass of the op."""
        # Remember the inputs from the forward pass
        # for the gradient calculations
        self.m1.v = first
        self.m2.v = second

        return first * second

    def backward_first(self, first: np.ndarray) -> np.ndarray:
        """Implement the backward pass of first tensor."""
        return np.ones_like(first) * self.m2.v

    def backward_second(self, second: np.ndarray) -> np.ndarray:
        """Implement the backward pass of second tensor."""
        return self.m1.v * np.ones_like(second)
