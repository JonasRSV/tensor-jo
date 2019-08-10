"""This module defines the nodes."""
import tensorjo
import numpy as np
import logging
from abc import abstractmethod
from . import op as operator
from . import math

LOGGER = logging.Logger(__name__)


class node():
    """All nodes are monoids or primitives under tensors and ops."""

    def __init__(self, name: str):
        """Store results of gradient calculations.

        When calculating the gradient for one tensor the gradient for
        all tensors which that tensor affect will also be calculated.

        To avoid doing many dubble calculations we store the gradient
        calculations throughout the backprops so that only gradients have to
        be calculated once per backprop.
        """
        self.name = name
        self.gradient = None
        self.gradient_cached = False
        self.output_cached = False
        self.output_cache = None

    @abstractmethod
    def output(self) -> np.ndarray:
        """Propagate value through node."""
        pass

    @abstractmethod
    def shape(self) -> tuple:
        """Return the shape of the output of the node."""
        pass

    def gradient_wrt(self, n: "node") -> np.ndarray:
        """Calculate the gradient wrt n.

        This is super central.

        This uses the chain rule as follows:

        Notation:

        c_node = output of current node
        n_i_node = output of next node
        err = the error we are calculating the gradient wrt
        using d as prefix means derivative.

        derr / dc_node = sum_{i} (derr / dn_i_node) * (dn_i_node / dc_node)

        In words:
        The gradient with respect to the current nodes output is:
        'The sum of the contributions to the next nodes times the next nodes
        contribution to the error'
        """
        # Base case
        if self == n:
            return np.array(1, dtype=np.float32)

        if self.gradient_cached:
            return self.gradient

        self.gradient = np.zeros(self.shape())
        for con in self.c:
            con_wrt_n = con.n.gradient_wrt(n)
            self_wrt_con = con.gradient_op()

            # This might be the most important line of all in this program
            self.gradient = self.gradient + (con_wrt_n * self_wrt_con)

        self.gradient_cached = True
        return self.gradient

    def __add__(self, other):
        """Add add op to graph."""
        return math.add(self, other)

    def __radd__(self, other):
        """Add add op to graph."""
        return math.add(other, self)

    def __iadd__(self, other):
        """Add add op to graph."""
        return math.add(self, other)

    def __sub__(self, other):
        """Add sub op to graph."""
        return math.sub(self, other)

    def __rsub__(self, other):
        """Add sub op to graph."""
        return math.sub(other, self)

    def __isub__(self, other):
        """Add sub op to graph."""
        return math.sub(self, other)

    def __mul__(self, other):
        """Add mult op to graph."""
        return math.mul(self, other)

    def __rmul__(self, other):
        """Add mult op to graph."""
        return math.mul(other, self)

    def __imul__(self, other):
        """Add mult op to graph."""
        return math.mul(self, other)

    def __truediv__(self, other):
        """Add div op to graph."""
        return math.div(self, other)

    def __rtruediv__(self, other):
        """Add div op to graph."""
        return math.div(other, self)

    def __idiv__(self, other):
        """Add div op to graph."""
        return math.div(self, other)


class connection():
    """Connections between node to add information.

    The information is which gradient op the node
    should call to receive its gradients.
    """

    def __init__(self, n: node, gradient_op):
        """Initialize the connection with the nodes and gradient op."""
        self.n: node = n
        self.gradient_op = gradient_op


class primitive(node):
    """Primitive node acts as entry to graph."""

    def __init__(self, v: np.ndarray, name):
        """Primitive node consists of only a tensor."""
        super().__init__(name)
        self.v: np.ndarray = v
        self.c: [connection] = []
        """Initially nodes are not cached unless a user calls cache on the graph.

        So that things can become pre-computed.
        (e.g calculation paths and so on)
        """
        self.update = self._no_cache_update
        """The all nodes depending on this node.

        All nodes in the calculation dependencies needs to have their cache
        emptied if this node is altered.

        The calculation path will be updated by the graph object whenever
        a user calls to initialize the cache. The graph object will also
        overload the output function of this object with its "_output_cache"

        The graph object is also responsible for reverting the cache if the
        user asks for it.
        """
        self.calculation_dependencies = []

    def output(self) -> np.ndarray:
        """Return the np.ndarray."""
        self.gradient_cached = False

        return self.v

    def shape(self) -> tuple:
        """Return shape of primitive np.ndarray."""
        return self.v.shape

    def _no_cache_update(self, v) -> node:
        """Update the underlying array."""
        v = np.array(v, dtype=np.float32)

        if self.v.shape != v.shape:
            raise ValueError("Cannot update tensor of shape %s with shape %s" %
                             (self.v.shape, v.shape))

        self.v = v

        return self

    def _cache_update(self, v) -> node:
        """Update the underlying array."""
        v = np.array(v, dtype=np.float32)

        if self.v.shape != v.shape:
            raise ValueError("Cannot update tensor of shape %s with shape %s" %
                             (self.v.shape, v.shape))

        self.v = v
        """Empty all caches."""
        for node in self.calculation_dependencies:
            node.output_cached = False

        return self

    def update(self) -> np.ndarray:
        """One of "_cache_update or _no_cache_update"."""
        raise NotImplementedError("update not implemented for primitive.")

    def __str__(self):
        """Return string rep of underlying array."""
        return str(self.v)


class monoid(node):
    """Monoid node is a building block of the graph."""

    def __init__(self, m1: node, m2: node, op: operator.Op, name: str = None):
        """Monoid: two elements and the binary operator."""
        if name is None:
            super().__init__(tensorjo.naming.get_node_name(op.name()))
        else:
            super().__init__(name)

        self.m1: node = m1
        self.m2: node = m2
        self.op: operator.Op = op
        """Forward connections."""
        self.c: [connection] = []
        """Initially nodes are not cached unless a user calls cache on the graph.

        So that things can become pre-computed.
        (e.g calculation paths and so on)
        """
        self.output = self._output_no_cache

    def _output_no_cache(self) -> np.ndarray:
        """Apply op on the inputs."""
        self.gradient_cached = False

        return self.op.forward(self.m1.output(), self.m2.output())

    def _output_cache(self) -> np.ndarray:
        """Apply op on inputs if output is not cached."""
        self.gradient_cached = False

        if self.output_cached:
            return self.output_cache

        self.output_cached = True
        self.output_cache = self.op.forward(self.m1.output(), self.m2.output())

        return self.output_cache

    def output(self) -> np.ndarray:
        """One of "output_no_cache or _output_cache"."""
        raise NotImplementedError("output not implemented for monoid.")

    def shape(self) -> tuple:
        """Return shape of monoid operator output."""
        return self.op.shape()


class functor(node):
    """functor node is a base building block of the graph."""

    def __init__(self, m1: node, op: operator.Op, name: str = None):
        """Monoid: two elements and the binary operator."""
        if name is None:
            super().__init__(tensorjo.naming.get_node_name(op.name()))
        else:
            super().__init__(name)

        self.m1: node = m1
        self.op: operator.Op = op
        """Forward connections."""
        self.c: [connection] = []
        """Initially nodes are not cached unless a user calls cache on the graph.

        So that things can become pre-computed.
        (e.g calculation paths and so on)
        """
        self.output = self._output_no_cache

    def _output_no_cache(self) -> np.ndarray:
        """Apply op on the inputs."""
        self.gradient_cached = False

        return self.op.forward(self.m1.output())

    def _output_cache(self) -> np.ndarray:
        """Apply op on inputs if output is not cached."""
        self.gradient_cached = False

        if self.output_cached:
            return self.output_cache

        self.output_cached = True
        self.output_cache = self.op.forward(self.m1.output())

        return self.output_cache

    def output(self) -> np.ndarray:
        """One of "output_no_cache or _output_cache"."""
        raise NotImplementedError("output not implemented for functor.")

    def shape(self) -> tuple:
        """Return shape of monoid operator output."""
        return self.op.shape()
