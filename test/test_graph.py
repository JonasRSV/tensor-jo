"""Tensor module."""
import tensorjo as tj
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)


def test_adding_variables():
    """See if variables is added to graph."""
    LOGGER.info("Adding variables.")
    tj.tjgraph.clear()
    tj.var(np.random.rand())
    tj.var(np.random.rand(1, 10))
    tj.var(np.random.rand(5, 5))
    tj.var(np.random.rand(), name="Hello")

    LOGGER.info("Getting variables.")
    vars = tj.tjgraph.get_variables()
    assert len(vars) == 4, "Variables should be len 4 not %s" % len(vars)
    assert "Hello" in [v.name for v in vars],\
        "Hello should be in vars found: %s" % [v.name for v in vars]

    LOGGER.info("Clearing variables.")
    tj.tjgraph.clear()

    vars = tj.tjgraph.get_variables()
    assert len(vars) == 0, "Cleared graph contained %s vars" % len(vars)

    tj.var(np.random.rand())
    tj.var(np.random.rand(1, 10))
    tj.var(np.random.rand(5, 5))
    tj.var(np.random.rand(), name="Hello")

    LOGGER.info("Getting specific variable.")
    vars = tj.tjgraph.get_variables(names=["Hello"])
    assert len(vars) == 1,\
        "Should only be on variable named 'Hello' found %s"\
        % [v.name for v in vars]

    assert "Hello" in [v.name for v in vars],\
        "'Hello' should be in %s" % [v.name for v in vars]

    LOGGER.info("Create two variables with same name: 'Hi' and 'Hi'")
    v1 = tj.var(np.random.rand(), name="Hi")
    v2 = tj.var(np.random.rand(), name="Hi")

    assert v1.name != v2.name,\
        "name collision %s and %s" % (v1.name, v2.name)

    LOGGER.info("Name after creation: %s and %s" % (v1.name, v2.name))

    tj.tjgraph.clear()


def test_adding_monoids():
    """See if monoids is added to graph."""
    tj.tjgraph.clear()

    LOGGER.info("Checking right number of nodes.")
    a = tj.var(np.random.rand(1, 5))
    b = tj.var(np.random.rand(1, 5))

    c = tj.add(a, b, name="first-addition")

    nodes = tj.tjgraph.get_nodes()
    vars = tj.tjgraph.get_variables()

    assert len(nodes) == 3, "Nodes should be 3 is %s" % len(nodes)
    assert len(vars) == 2, "Vars should be 2 is %s" % len(vars)

    LOGGER.info("Checking names.")
    var_names = [v.name for v in vars]
    node_names = [n.name for n in nodes]
    assert "first-addition" in node_names,\
        "first-addition should be in %s" % node_names

    assert "first-addition" not in var_names,\
        "first-addition should not be in %s" % var_names

    d = tj.mul(c, b, name="first-multiplication")

    nodes = tj.tjgraph.get_nodes()
    node_names = [n.name for n in nodes]

    LOGGER.info("Checking more right number of nodes.")
    assert len(nodes) == 4, "Nodes should be 4 is %s" % len(nodes)
    assert "first-addition" in node_names,\
        "first-addition should be in %s" % node_names

    tj.tjgraph.clear()

    nodes = tj.tjgraph.get_nodes()
    vars = tj.tjgraph.get_variables()

    LOGGER.info("Testing clearing")
    assert len(nodes) == 0, "Nodes should be 0 is %s" % len(nodes)
    assert len(vars) == 0, "Vars should be 0 is %s" % len(vars)

    a = tj.var(np.random.rand(1, 5))
    b = tj.var(np.random.rand(1, 5))

    c = tj.add(a, b)
    c = tj.add(c, a)
    c = tj.add(c, b)
    d = tj.mul(c, b, name="cookie")
    e = tj.sub(c, d)
    f = tj.div(d, e, name="milk")
    f = tj.add(f, f)

    nodes = tj.tjgraph.get_nodes()
    vars = tj.tjgraph.get_variables()

    var_names = [v.name for v in vars]
    node_names = [n.name for n in nodes]

    LOGGER.info("Test adding a lot of monoids")
    assert len(nodes) == 9, "Nodes should be 9 is %s" % len(nodes)
    assert len(vars) == 2, "Vars should be 2 is %s" % len(vars)

    LOGGER.info("Checking names")
    assert "cookie" in node_names,\
        "cookie should be in %s" % node_names

    assert "cookie" not in var_names,\
        "cookie should not be in %s" % var_names

    assert "milk" in node_names,\
        "milk should be in %s" % node_names

    assert "milk" not in var_names,\
        "milk should not be in %s" % var_names

    tj.tjgraph.clear()

    nodes = tj.tjgraph.get_nodes()
    vars = tj.tjgraph.get_variables()

    LOGGER.info("Testing clearing")
    assert len(nodes) == 0, "Nodes should be 0 is %s" % len(nodes)
    assert len(vars) == 0, "Vars should be 0 is %s" % len(vars)


def test_adding_functors():
    """See if monoids is added to graph."""
    tj.tjgraph.clear()

    LOGGER.info("Checking right number of nodes.")
    a = tj.var(np.random.rand(1, 5))

    c = tj.sigmoid(a, name="sigmoid")

    nodes = tj.tjgraph.get_nodes()
    vars = tj.tjgraph.get_variables()

    assert len(nodes) == 2, "Nodes should be 3 is %s" % len(nodes)
    assert len(vars) == 1, "Vars should be 2 is %s" % len(vars)

    LOGGER.info("Checking names.")
    var_names = [v.name for v in vars]
    node_names = [n.name for n in nodes]
    assert "sigmoid" in node_names,\
        "sigmoid should be in %s" % node_names

    assert "sigmoid" not in var_names,\
        "sigmoid should not be in %s" % var_names

    d = tj.sigmoid(c, name="sig")

    nodes = tj.tjgraph.get_nodes()
    node_names = [n.name for n in nodes]

    LOGGER.info("Checking more right number of nodes.")
    assert len(nodes) == 3, "Nodes should be 3 is %s" % len(nodes)
    assert "sig" in node_names,\
        "sig should be in %s" % node_names

    tj.tjgraph.clear()

    nodes = tj.tjgraph.get_nodes()
    vars = tj.tjgraph.get_variables()

    LOGGER.info("Testing clearing")
    assert len(nodes) == 0, "Nodes should be 0 is %s" % len(nodes)
    assert len(vars) == 0, "Vars should be 0 is %s" % len(vars)

    a = tj.var(np.random.rand(1, 5))
    b = tj.var(np.random.rand(1, 5))

    c = tj.add(a, b)
    c = tj.add(c, a)
    c = tj.add(c, b)
    d = tj.mul(c, b, name="cookie")
    e = tj.sub(c, d)
    f = tj.div(d, e, name="milk")
    f = tj.add(f, f)
    f = tj.sigmoid(f)

    nodes = tj.tjgraph.get_nodes()
    vars = tj.tjgraph.get_variables()

    var_names = [v.name for v in vars]
    node_names = [n.name for n in nodes]

    LOGGER.info("Test adding a lot of monoids and a functor")
    assert len(nodes) == 10, "Nodes should be 10 is %s" % len(nodes)
    assert len(vars) == 2, "Vars should be 2 is %s" % len(vars)

    LOGGER.info("Checking names")
    assert "cookie" in node_names,\
        "cookie should be in %s" % node_names

    assert "cookie" not in var_names,\
        "cookie should not be in %s" % var_names

    assert "milk" in node_names,\
        "milk should be in %s" % node_names

    assert "milk" not in var_names,\
        "milk should not be in %s" % var_names

    tj.tjgraph.clear()

    nodes = tj.tjgraph.get_nodes()
    vars = tj.tjgraph.get_variables()

    LOGGER.info("Testing clearing")
    assert len(nodes) == 0, "Nodes should be 0 is %s" % len(nodes)
    assert len(vars) == 0, "Vars should be 0 is %s" % len(vars)
