"""This module defines the graph. TODO Some errors in this file with gradient calculations"""
from abc import abstractmethod
from tensorjo import tensor
from . import op as operator
import typecheck as tc
import numpy as np


class node():
    """All nodes are monoids or primitives under tensors and ops."""

    @abstractmethod
    def output(self) -> tensor:
        """Propagate value through node."""
        pass

    def gradient_wrt(self, n: "node") -> np.ndarray:
        """Calculate the gradient wrt n."""
        gradient = np.zeroes_like(self.t)

        for con in self.c:
            con_wrt_n = con.t.gradient_wrt(n)
            self_wrt_con = con.gradient_op(self.t)

            # Chain rule
            gradient += con_wrt_n * self_wrt_con

        return gradient


class connection():
    """Connections between node to add information.

    The information is which gradient op the node
    should call to receive its gradients.
    """

    def __init__(self, t: node, gradient_op):
        """Initialize the connection with the nodes and gradient op."""
        self.t: node = t
        self.gradient_op = gradient_op


class primitive(node):
    """Primitive node acts as entry to graph."""

    @tc.typecheck
    def __init__(self, t: tensor):
        """Primitive node consists of only a tensor."""
        super()
        self.t: tensor = t
        self.c: [connection] = []

    def output(self) -> tensor:
        """Return the tensor."""
        return self.t


class monoid(node):
    """Monoid node is the base building block of the graph."""

    @tc.typecheck
    def __init__(self, m1: node, m2: node, op: operator.Op):
        """Monoid: two elements and the binary operator."""
        super()
        self.m1: node = m1
        self.m2: node = m2
        self.op: operator.Op = op
        """Forward connections."""
        self.c: [connection] = []

    def output(self) -> np.ndarray:
        """Apply op on the inputs."""
        return self.op.forward(self.m1.output(), self.m2.output())


@tc.typecheck
def apply(m1: node, m2: node, op: operator.Op) -> node:
    """Graph building op.

    This op is responsible for making the correct connections
    """
    m = monoid(m1, m2, op)
    m1.c.append(connection(m, op.backward_first))
    m2.c.append(connection(m, op.backward_second))
    return m
