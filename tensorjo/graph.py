"""This module defines the graph.

The graph keeps track of all the nodes and are responsible
for fetching variables and controlling the cache.

The graph is also responsible for adding stuff.
"""
import tensorjo
from . import op as operator
from . import node
import numpy as np
import logging

LOGGER = logging.Logger(__name__)


class graph():
    """Keep track of all nodes and provides some utilities."""

    def __init__(self, name: str):
        """Initialize the graph."""
        self.name = name
        self.nodes = {}

        # variables are also contained in nodes
        # This exists because variables are probably
        # going to be requested often for gradient calculations
        self.variables = {}

    def get_variables(self, names: [str] = None):
        """Return the variables in the names list."""
        if names is None:
            return list(self.variables.values())

        vars = []
        for name in names:
            if name in self.variables:
                vars.append(self.variables[name])
            else:
                raise ValueError("%s is not a variable in the graph %s" %
                                 (name, self.name))

        return vars

    def get_nodes(self, names: [str] = None):
        """Return the nodes in the names list."""
        if names is None:
            return list(self.nodes.values())

        nodes = []
        for name in names:
            if name in self.nodes:
                nodes.append(self.nodes[name])
            else:
                raise ValueError("%s is not a node in the graph %s" %
                                 (name, self.name))

        return nodes

    def add(self, n: "node.node", variable=False):
        """Add a node to the graph."""
        while n.name in self.nodes:
            nn = tensorjo.naming.get_node_name(n.name)
            LOGGER.warning("%s is already in the graph renaming it to: %s" %
                           (n.name, nn))
            n.name = nn

        self.nodes[n.name] = n

        if variable:
            self.variables[n.name] = n

    def clear(self):
        """Clear the entire graph."""
        self.nodes = {}
        self.variables = {}

    def cache(self):
        """Make computations cached in graph.

        This requires a bit of precomputation and is therefore
        not on by default.

        If the user adds ops after calling cache then cache needs
        to be called again.
        """
        for n in self.nodes.values():
            if isinstance(n, node.primitive):
                n.calculation_dependencies = get_calculation_dependencies(n)
                n.update = n._cache_update
            else:
                n.output = n._output_cache

    def no_cache(self):
        """Make computations uncached."""
        for n in self.nodes.values():
            if isinstance(n, node.primitive):
                n.update = n._no_cache_update
            else:
                n.output = n._output_no_cache


"""Define some graph utilities."""


def get_calculation_dependencies(node: "node.node") -> ["node.node"]:
    """Get the calculation dependencies of a node.

    All nodes whos output depends on the output of this node.

    There can be circles so need to be ware of them.
    """
    mem = set()

    def dfs(n: "node.node"):
        """DFS the graph to find all connected nodes."""
        if n in mem:
            return

        mem.add(n)

        for c in n.c:
            dfs(c.t)

    dfs(node)
    return list(mem)


def apply_monoid(m1: "node.node",
                 m2: "node.node",
                 op: operator.Op,
                 name: str = None) -> "node.node":
    """Graph building op.

    This op is responsible for making the correct connections
    """
    m1_c = np.ones(m1.shape(), dtype=np.float32)
    m2_c = np.ones(m2.shape(), dtype=np.float32)

    init_op = op(m1_c, m2_c)

    m = node.monoid(m1, m2, init_op, name=name)
    m1.c.append(node.connection(m, init_op.backward_first))
    m2.c.append(node.connection(m, init_op.backward_second))
    """Add node to graph."""
    tensorjo.tjgraph.add(m)

    return m


def apply_functor(m1: "node.node", op: operator.Op,
                  name: str = None) -> "node.node":
    """Graph building op.

    This op is responsible for making the correct connections
    """
    m1_c = np.ones(m1.shape(), dtype=np.float32)

    init_op = op(m1_c)

    m = node.functor(m1, init_op, name=name)
    m1.c.append(node.connection(m, init_op.backward_functor))
    """Add node to graph."""
    tensorjo.tjgraph.add(m)

    return m
