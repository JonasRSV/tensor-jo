"""This files defines the normal addition op."""
from tensorjo import op
from tensorjo import tensor
import typecheck as tc
import numpy as np


class addition(op.Op):
    """This class implements the forward and backward pass for addition."""

    @tc.typecheck
    def __init__(self, m1: tensor, m2: tensor):
        """Initialize op.

        1. Make sure the two tensors can have this op applied to them.
        """
        pass

    def forward(self, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        """Implement the forward pass of the op."""
        pass

    def backward_first(self, first: np.ndarray) -> np.ndarray:
        """Implement the backward pass of first tensor."""
        pass

    def backward_second(self, second: np.ndarray) -> np.ndarray:
        """Implement the backward pass of second tensor."""
        pass
