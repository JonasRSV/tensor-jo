"""This module defines the structure of the op in the graph."""
from abc import abstractmethod
import numpy as np


class optimiser():
    """An optimiser for ops in the graph.

    The optimiser should be initialized with the node to optimise.

    Then the optimiser should optimise minimize and maximize ops
    """

    @abstractmethod
    def maximise(self, nodes: ["node.node"]) -> [np.ndarray]:
        """Maximise self with respect to the nodes."""
        raise NotImplementedError("maximise is not implemented.")

    @abstractmethod
    def minimise(self, nodes: ["node.node"]) -> [np.ndarray]:
        """Minimise self with respect to the nodes."""
        raise NotImplementedError("minimise is not implemented.")
