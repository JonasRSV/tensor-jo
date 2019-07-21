"""Vanilla gradient optimiser module."""
import tensorjo
from tensorjo import optimiser
import numpy as np


class gd(optimiser.optimiser):
    """Gradient Optimiser."""

    def __init__(self, master: "node.node"):
        """Initialise the optimiser with node optimising against."""
        super()

        self.master = master
        """update step size."""
        self.dt = 1e-2

        """Rounds to optimise."""
        self.rounds = 100

    def maximise(self, nodes: ["node.node"]) -> None:
        """Maximise op."""
        cache = list(enumerate(nodes))
        for i in range(self.rounds):
            grads = tensorjo.gradients(self.master, nodes)
            for i, n in cache:
                n.update(n.v + np.mean(grads[i]) * self.dt)

    def minimise(self, nodes: ["node.node"]) -> None:
        """Minimise op."""
        cache = list(enumerate(nodes))
        for i in range(self.rounds):
            grads = tensorjo.gradients(self.master, nodes)
            for i, n in cache:
                n.update(n.v - np.mean(grads[i]) * self.dt)
