"""This files defines the normal multiplication op."""
from tensorjo import op
import numpy as np


class multiplication(op.Op):
    """Implements the forward and backward pass for multiplication."""

    def __init__(self, m1: np.ndarray, m2: np.ndarray):
        """Initialize op."""
        super()

        self.output_shape = None
        try:
            self.output_shape = (m1 * m2).shape
        except ValueError as e:
            raise ValueError(
                "Failed to construct multiplication op with tensors %s and %s "
                % (m1, m2) + "- %s" % e)

        self.m1 = m1
        self.m2 = m2

        self.c = m1 * m2

    def forward(self, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        """Implement the forward pass of the op."""
        # Remember the inputs from the forward pass
        # for the gradient calculations
        self.m1 = m1
        self.m2 = m2
        self.c = m1 * m2
        return self.c

    def backward_first(self) -> np.ndarray:
        """Implement the backward pass of first tensor."""
        return np.ones_like(self.m1) * self.m2

    def backward_second(self) -> np.ndarray:
        """Implement the backward pass of second tensor."""
        return self.m1 * np.ones_like(self.m2)

    def cache(self) -> np.ndarray:
        """Return output from previous forward pass."""
        return self.c

    def shape(self):
        """Return the shape of the forward pass."""
        return self.output_shape

    def name(self):
        """Return name of multiplication op."""
        return "multiplication"
