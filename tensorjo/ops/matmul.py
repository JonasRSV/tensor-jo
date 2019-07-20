"""This files defines the normal multiplication op."""
from tensorjo import op
import numpy as np


class matmul(op.Op):
    """Implements the forward and backward pass for matmul."""

    def __init__(self, m1: np.ndarray, m2: np.ndarray):
        """Initialize op."""
        self.output_shape = None
        try:
            self.output_shape = (np.matmul(m1, m2)).shape
        except ValueError as e:
            raise ValueError(
                "Failed to construct matmul op with tensors %s and %s " %
                (m1, m2) + "- %s" % e)

        self.m1 = m1
        self.m2 = m2

        self.c = m1 @ m2

    def forward(self, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        """Implement the forward pass of the op."""
        # Remember the inputs from the forward pass
        # for the gradient calculations
        self.m1 = m1
        self.m2 = m2
        self.c = np.matmul(m1, m2)

        return self.c

    def backward_first(self) -> np.ndarray:
        """Implement the backward pass of first tensor."""
        # TODO:
        raise NotImplementedError()

    def backward_second(self) -> np.ndarray:
        """Implement the backward pass of second tensor."""
        # TODO:
        raise NotImplementedError()

    def cache(self) -> np.ndarray:
        """Return output from previous forward pass."""
        return self.c

    def shape(self):
        """Return the shape of the forward pass."""
        return self.output_shape

    def name(self):
        """Return name of addition op."""
        return "matmul"
