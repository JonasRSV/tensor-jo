"""This files defines the normal multiplication op."""
from tensorjo import op
import numpy as np


class division(op.Op):
    """Implements the forward and backward pass for division."""

    # To avoid zero divison
    tiny_number = 1e-15

    def __init__(self, m1: np.ndarray, m2: np.ndarray):
        """Initialize op."""
        super()

        self.output_shape = None
        try:
            self.output_shape = m1 / (m2 + division.tiny_number)
        except ValueError as e:
            raise ValueError(
                "Failed to construct division op with tensors %s and %s " %
                (m1, m2) + "- %s" % e)

        self.m1 = m1
        self.m2 = m2

        self.c = m1 / (m2 + division.tiny_number)

    def forward(self, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        """Implement the forward pass of the op."""
        # Remember the inputs from the forward pass
        # for the gradient calculations
        self.m1 = m1
        self.m2 = m2
        self.c = m1 / (m2 + division.tiny_number)
        return self.c

    def backward_first(self) -> np.ndarray:
        """Implement the backward pass of first tensor."""
        return np.ones_like(self.m1) / (self.m2 + division.tiny_number)

    def backward_second(self) -> np.ndarray:
        """Implement the backward pass of second tensor."""
        return -self.m1 / (self.m2 * self.m2 + division.tiny_number)

    def cache(self) -> np.ndarray:
        """Return output from previous forward pass."""
        return self.c

    def shape(self):
        """Return the shape of the forward pass."""
        return (self.m1 / (self.m2 + division.tiny_number)).shape

    def name(self):
        """Return name of division op."""
        return "division"
