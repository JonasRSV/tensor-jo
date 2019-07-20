"""This files defines the MSE op."""
from tensorjo import op
import numpy as np


class mse(op.Op):
    """This class implements the forward and backward pass for mse."""

    def __init__(self, m1: np.ndarray, m2: np.ndarray):
        """Initialize op."""
        super()

        self.output_shape = None
        try:
            self.output_shape = np.mean(np.square(m1 - m2), axis=1).shape
        except (ValueError, IndexError) as e:
            raise ValueError(
                "Failed to construct mse op with tensors %s and %s " %
                (m1, m2) + "- %s" % e)

        self.m1 = m1
        self.m2 = m2

        self.c = np.mean(np.square(m1 - m2), axis=1)

    def forward(self, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        """Implement the forward pass of the op."""
        # Remember the inputs from the forward pass
        # for the gradient calculations
        self.m1 = m1
        self.m2 = m2
        self.c = np.mean(np.square(m1 - m2), axis=1)

        return self.c

    def backward_first(self) -> np.ndarray:
        """Implement the backward pass of first tensor."""
        difference = self.m1 - self.m2
        return 2 * difference / len(difference)

    def backward_second(self) -> np.ndarray:
        """Implement the backward pass of second tensor."""
        difference = self.m1 - self.m2
        return -2 * difference / len(difference)

    def cache(self) -> np.ndarray:
        """Return output from previous forward pass."""
        return self.c

    def shape(self):
        """Return the shape of the forward pass."""
        return self.output_shape

    def name(self):
        """Return name of mse op."""
        return "mse"
