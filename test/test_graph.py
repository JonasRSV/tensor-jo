"""Tensor module."""
from tensorjo import viz
import tensorjo as tj
import numpy as np
import logging
import time
import sys

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


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


def test_cache():
    """Test the cache functionality of the graph."""
    LOGGER.info("Testing cache.")

    LOGGER.info("Running logistic regression test.")

    x = np.random.rand(10) - 0.5
    y = sigmoid(4 * x - 1)

    a = tj.var(np.random.rand())
    b = tj.var(np.random.rand())

    o = tj.sigmoid(a * x + b)
    err = tj.mse(y, o)

    LOGGER.info("Optimizing a simple logistic regression.")
    LOGGER.info(" X domain is (10, 2)")
    LOGGER.info(" Y domain is (10, 1)")

    LOGGER.info("before training: coefficient %s -- bias: %s -- mse: %s" %
                (a, b, err.output()))

    opt = tj.opt.gd(err)
    opt.dt = 1e-0
    opt.rounds = 5000

    opt.minimise([a, b])

    LOGGER.info("after training: coefficient %s -- bias: %s -- mse: %s" %
                (a, b, err.output()))
    """ ------------ """

    LOGGER.info("Enabling cache and running the same calculations.")

    tj.tjgraph.cache()

    a.update(np.random.rand())
    b.update(np.random.rand())

    LOGGER.info("before training: coefficient %s -- bias: %s -- mse: %s" %
                (a, b, err.output()))

    opt = tj.opt.gd(err)
    opt.dt = 1e-0
    opt.rounds = 5000

    opt.minimise([a, b])

    LOGGER.info("after training: coefficient %s -- bias: %s -- mse: %s" %
                (a, b, err.output()))
    """ ------------ """

    LOGGER.info("Disableing cache and running the same calculations.")

    tj.tjgraph.no_cache()

    a.update(np.random.rand())
    b.update(np.random.rand())

    LOGGER.info("before training: coefficient %s -- bias: %s -- mse: %s" %
                (a, b, err.output()))

    opt = tj.opt.gd(err)
    opt.dt = 1e-0
    opt.rounds = 5000

    opt.minimise([a, b])

    LOGGER.info("after training: coefficient %s -- bias: %s -- mse: %s" %
                (a, b, err.output()))
    """ ------------ """

    LOGGER.info("Testing if cache makes a difference performance wise.")
    """Make a loong graph."""
    a = tj.var(np.random.rand())
    b = tj.var(np.random.rand())

    sys.setrecursionlimit(5000)

    timestamp = time.time()
    c = a + b
    for _ in range(2000):
        c = a + b + c

    LOGGER.info("Making graph with %s ops took %s seconds" %
                (3 * 2000, time.time() - timestamp))

    iters = 200

    timestamp = time.time()
    for _ in range(iters):
        c.output()

    LOGGER.info("Running %s iters without cache took %s seconds" %
                (iters, time.time() - timestamp))

    timestamp = time.time()
    tj.tjgraph.cache()

    LOGGER.info("Cacheing graph took %s seconds" % (time.time() - timestamp))

    timestamp = time.time()
    for _ in range(iters):
        c.output()

    LOGGER.info("Running %s iters with cache took %s seconds" %
                (iters, time.time() - timestamp))

    timestamp = time.time()
    for _ in range(iters):
        a.update(np.random.rand())
        c.output()

    LOGGER.info("Running %s iters with cache and update took %s seconds" %
                (iters, time.time() - timestamp))

    tj.tjgraph.no_cache()

    timestamp = time.time()
    for _ in range(iters):
        a.update(np.random.rand())
        c.output()

    LOGGER.info("Running %s iters with no cache and update took %s seconds" %
                (iters, time.time() - timestamp))


def test_remove():
    """Test the graph remove functionality."""
    LOGGER.info("Testing removing a functor.")
    LOGGER.info("Building Graph.")

    tj.tjgraph.clear()

    a = tj.var(5)
    b = tj.sigmoid(a)
    c = tj.sigmoid(b)
    c = tj.sigmoid(c)

    d = c + 5
    d = d + 5

    v = viz.visualizer(tj.tjgraph)

    v.draw(d)

    assert len(tj.tjgraph.nodes) == 6, "Graph should contain 5 nodes "\
        + "Graph contains %s nodes" % len(tj.tjgraph.nodes)

    tj.tjgraph.remove(a)

    assert len(tj.tjgraph.nodes) == 1, "Graph should contain 1 monoid "\
        + "Graph contains %s nodes" % len(tj.tjgraph.nodes)

    o = d.output()
    assert d.output() == 10, ("Output should be 5 is %s" % o)

    v.draw(d)
