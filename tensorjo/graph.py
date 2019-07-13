"""This module defines the graph."""
from abc import abstractmethod
from tensorjo import tensor
from . import op as operator
import typecheck as tc


class node():
    """All nodes are monoids or primitives under tensors and ops."""

    @abstractmethod
    def output(self):
        """Propagate value through node."""
        pass


class primitive(node):
    """Primitive node acts as entry to graph."""

    def __init__(self, t: tensor):
        """Primitive node consists of only a tensor."""
        self.t = t


class monoid(node):
    """Monoid node is the base building block of the graph."""

    @tc.typecheck
    def __init__(self, m1: node, m2: node, op: operator.Op):
        """Monoid two nodes and the binary operator."""
        self.m1 = m1
        self.m2 = m2
        self.op = op
