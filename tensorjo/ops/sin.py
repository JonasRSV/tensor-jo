"""This files defines the normal sin op."""
from tensorjo import op
import numpy as np


def s(x):
    """Sin function."""
    return np.sin(x)


class sin(op.Op):
    """This class implements the forward and backward pass for sin."""

    def __init__(self, m1: np.ndarray):
        """Initialize op."""
        super()

        self.output_shape = None
        try:
            self.output_shape = s(m1).shape
        except ValueError as e:
            raise ValueError("Failed to construct sin op with tensor %s" %
                             (m1) + "- %s" % e)

        self.m1 = np.array(m1, copy=True)
        self.c = s(np.array(m1, copy=True))

    def forward(self, m1: np.ndarray) -> np.ndarray:
        """Implement the forward pass of the op."""
        self.m1 = m1
        self.c = s(m1)
        return self.c

    def backward_functor(self) -> np.ndarray:
        """Implement the backward pass of first tensor."""
        return np.cos(self.m1)

    def cache(self) -> np.ndarray:
        """Return output from previous forward pass."""
        return self.c

    def shape(self):
        """Return the shape of the forward pass."""
        return self.output_shape

    def name(self):
        """Return name of sin op."""
        return "sin"
