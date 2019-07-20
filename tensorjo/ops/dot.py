"""This files defines the normal dot op."""
from tensorjo import op
import numpy as np


class dot(op.Op):
    """This class implements the forward and backward pass for dot."""

    def __init__(self, m1: np.ndarray, m2: np.ndarray):
        """Initialize op."""
        super()

        self.output_shape = None
        try:
            self.output_shape = (m1 @ m2).shape
        except ValueError as e:
            raise ValueError(
                "Failed to construct dot op with tensors %s and %s " %
                (m1, m2) + "- %s" % e)

        self.m1 = m1
        self.m2 = m2

        if len(self.output_shape) < 2:
            raise ValueError(
                "dot product with tensor of dim < 2 is not supported.")

        self.c = m1 @ m2
        """Store a few arrays here to speed up calculations."""
        self.M = m1.shape[1]
        self.first_grad = np.ones_like(m1)

        self.N = m2.shape[0]
        self.second_grad = np.ones_like(m2)

    def forward(self, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        """Implement the forward pass of the op."""
        self.m1 = m1
        self.m2 = m2
        self.c = m1 @ m2
        return self.c

    def backward_first(self) -> np.ndarray:
        """Implement the backward pass of first tensor."""
        for i in range(self.M):
            self.first_grad[:, i] = np.sum(self.m2[i, :])
        return self.first_grad

    def backward_second(self) -> np.ndarray:
        """Implement the backward pass of second tensor."""
        for i in range(self.N):
            self.second_grad[i, :] = np.sum(self.m1[:, i])
        return self.second_grad

    def cache(self) -> np.ndarray:
        """Return output from previous forward pass."""
        return self.c

    def shape(self):
        """Return the shape of the forward pass."""
        return self.output_shape

    def name(self):
        """Return name of dot op."""
        return "dot"
