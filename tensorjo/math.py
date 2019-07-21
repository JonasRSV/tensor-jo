"""Define all math ops."""
import tensorjo
from tensorjo import ops
from . import node
from . import graph
import numpy as np


def ensure_node(m) -> "node.node":
    """Ensure that m1 & m2 are tensors or nodes."""
    if type(m) != node.node:
        m = tensorjo.tensor(m)

    return m


def add(m1, m2, name: str = None) -> "node.node":
    """Add add op to graph."""
    m1 = ensure_node(m1)
    m2 = ensure_node(m2)
    return graph.apply_monoid(m1, m2, ops.addition, name=name)


def sub(m1, m2, name: str = None) -> "node.node":
    """Add sub op to graph."""
    m1 = ensure_node(m1)
    m2 = ensure_node(m2)
    return graph.apply_monoid(m1, m2, ops.subtraction, name=name)


def mul(m1, m2, name: str = None) -> "node.node":
    """Add mul op to graph."""
    m1 = ensure_node(m1)
    m2 = ensure_node(m2)
    return graph.apply_monoid(m1, m2, ops.multiplication, name=name)


def div(m1, m2, name: str = None) -> "node.node":
    """Add div op to graph."""
    m1 = ensure_node(m1)
    m2 = ensure_node(m2)
    return graph.apply_monoid(m1, m2, ops.division, name=name)


def mse(m1, m2, name: str = None) -> "node.node":
    """Add mse op to graph."""
    m1 = ensure_node(m1)
    m2 = ensure_node(m2)
    return graph.apply_monoid(m1, m2, ops.mse, name=name)


def mean(m, axis: int, name: str) -> "node.node":
    """Calculate mean around axis."""
    if type(axis) != int:
        raise ValueError("Invalid axis type: got %s expected int" %
                         (type(axis)))

    m = ensure_node(m)
    return graph.apply_functor(m, lambda x: ops.mean(x, axis), name=name)


def sigmoid(m, name: str = None) -> "node.node":
    """Add sigmoid op to graph."""
    m = ensure_node(m)
    return graph.apply_functor(m, ops.sigmoid, name=name)


def sin(m, name: str = None) -> "node.node":
    """Add sin op to graph."""
    m = ensure_node(m)
    return graph.apply_functor(m, ops.sin, name=name)


def cos(m, name: str = None) -> "node.node":
    """Add cos op to graph."""
    m = ensure_node(m)
    return graph.apply_functor(m, ops.cos, name=name)


def var(obj, name: str = None) -> "node.node":
    """Create a variable."""
    node = tensorjo.tensor(obj, name=name)
    """Add node to graph."""
    tensorjo.tjgraph.add(node, variable=True)

    return node


def gradients(node: "node.node", primitives: ["node.node"]) -> [np.ndarray]:
    """Get gradients of the primitives with respect to the node."""
    # Propagate the graph once (To update the state)
    node.output()

    # If a gradient of a node is not connected to the 'node'
    # then the gradient will be 0
    # TODO: decide wheter that should throw error or not

    gradients = []
    for n in primitives:
        gradients.append(n.gradient_wrt(node))

    return gradients
