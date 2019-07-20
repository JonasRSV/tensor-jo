"""Tensor module."""
import tensorjo as tj
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)


def _true(item):
    try:
        return all(np.array(item).reshape(-1))
    except Exception as e:
        return item


def test_0d_simples():
    """Test trivial gradient updates."""
    LOGGER.info("Testing Addition.")
    a = tj.var(5)
    b = tj.var(10)

    c = tj.add(a, b)

    for i in range(5):
        g = tj.gradients(c, [a])
        LOGGER.info("%s *** a: %s -- b: %s -- g: %s -- dt: 0.1" % (i, a, b, g))

        a.update(a.v + g[0] * 1e-1)

    assert a.v > 5.0, "A should be larger than 5 but is %s" % a

    LOGGER.info("Testing Subtraction.")
    a = tj.var(5)
    b = tj.var(10)

    c = tj.sub(a, b)

    for i in range(5):
        g = tj.gradients(c, [b])
        LOGGER.info("%s *** a: %s -- b: %s -- g: %s -- dt: 0.1" % (i, a, b, g))

        b.update(b.v + g[0] * 1e-1)

    assert b.v < 10.0, "A should be smaller than 10 but is %s" % a

    LOGGER.info("Testing Multiplication.")
    a = tj.var(5)
    b = tj.var(10)

    c = tj.mul(a, b)

    for i in range(5):
        g = tj.gradients(c, [a])
        LOGGER.info("%s *** a: %s -- b: %s -- g: %s -- dt: 0.1" % (i, a, b, g))

        a.update(a.v + g[0] * 1e-1)

    assert a.v > 5.0, "A should be larger than 5 but is %s" % a

    LOGGER.info("Testing Division.")
    a = tj.var(5)
    b = tj.var(10)

    c = tj.div(a, b)

    for i in range(5):
        g = tj.gradients(c, [b])
        LOGGER.info("%s *** a: %s -- b: %s -- g: %s -- dt: 0.1" % (i, a, b, g))

        b.update(b.v + g[0] * 1e-1)

    assert b.v < 10.0, "B should be smaller than 10 but is %s" % a


def test_1d_simples():
    """Test trivial gradient updates."""
    LOGGER.info("Testing Addition.")
    a = tj.var(np.ones(3) * 5)
    b = tj.var(np.ones(3) * 10)

    c = tj.add(a, b)

    for i in range(5):
        g = tj.gradients(c, [a])
        LOGGER.info("%s *** a: %s -- b: %s -- g: %s -- dt: 0.1" % (i, a, b, g))

        a.update(a.v + g[0] * 1e-1)

    assert _true(a.v > 5.0), "A should be larger than 5 but is %s" % a

    LOGGER.info("Testing Subtraction.")
    a = tj.var(np.ones(3) * 5)
    b = tj.var(np.ones(3) * 10)

    c = tj.sub(a, b)

    for i in range(5):
        g = tj.gradients(c, [b])
        LOGGER.info("%s *** a: %s -- b: %s -- g: %s -- dt: 0.1" % (i, a, b, g))

        b.update(b.v + g[0] * 1e-1)

    assert _true(b.v < 10.0), "A should be smaller than 10 but is %s" % a

    LOGGER.info("Testing Multiplication.")
    a = tj.var(np.ones(3) * 5)
    b = tj.var(np.ones(3) * 10)

    c = tj.mul(a, b)

    for i in range(5):
        g = tj.gradients(c, [a])
        LOGGER.info("%s *** a: %s -- b: %s -- g: %s -- dt: 0.1" % (i, a, b, g))

        a.update(a.v + g[0] * 1e-1)

    assert _true(a.v > 5.0), "A should be larger than 5 but is %s" % a

    LOGGER.info("Testing Division.")
    a = tj.var(np.ones(3) * 5)
    b = tj.var(np.ones(3) * 10)

    c = tj.div(a, b)

    for i in range(5):
        g = tj.gradients(c, [b])
        LOGGER.info("%s *** a: %s -- b: %s -- g: %s -- dt: 0.1" % (i, a, b, g))

        b.update(b.v + g[0] * 1e-1)

    assert _true(b.v < 10.0), "B should be smaller than 10 but is %s" % a


def test_2d_simples():
    """Test trivial gradient updates."""
    LOGGER.info("Testing Addition.")
    a = tj.var(np.ones((3, 3)) * 5)
    b = tj.var(np.ones((3, 3)) * 10)

    c = tj.add(a, b)

    for i in range(5):
        g = tj.gradients(c, [a])

        a.update(a.v + g[0] * 1e-1)

    assert _true(a.v > 5.0), "A should be larger than 5 but is %s" % a

    LOGGER.info("Testing Subtraction.")
    a = tj.var(np.ones((3, 3)) * 5)
    b = tj.var(np.ones((3, 3)) * 10)

    c = tj.sub(a, b)

    for i in range(5):
        g = tj.gradients(c, [b])

        b.update(b.v + g[0] * 1e-1)

    assert _true(b.v < 10.0), "A should be smaller than 10 but is %s" % a

    LOGGER.info("Testing Multiplication.")
    a = tj.var(np.ones((3, 3)) * 5)
    b = tj.var(np.ones((3, 3)) * 10)

    c = tj.mul(a, b)

    for i in range(5):
        g = tj.gradients(c, [a])

        a.update(a.v + g[0] * 1e-1)

    assert _true(a.v > 5.0), "A should be larger than 5 but is %s" % a

    LOGGER.info("Testing Division.")
    a = tj.var(np.ones((3, 3)) * 5)
    b = tj.var(np.ones((3, 3)) * 10)

    c = tj.div(a, b)

    for i in range(5):
        g = tj.gradients(c, [b])

        b.update(b.v + g[0] * 1e-1)

    assert _true(b.v < 10.0), "B should be smaller than 10 but is %s" % a


def test_convex():
    """Test optimising a simple convex function."""
    LOGGER.info("Testing simple convex: x^2")
    for s in [1, 3]:
        a = tj.var(np.ones(s))
        b = tj.mul(a, a)

        LOGGER.info("Initial a: %s" % a)

        for i in range(100):
            g = tj.gradients(b, [a])

            a.update(a.v - g[0] * 1e-1)

        LOGGER.info("Final a: %s" % a)

        assert _true(
            abs(a.v) < 1.0), "A should be smaller than 1 but is %s" % a

    LOGGER.info("Testing more complex convex: (x * 5 + 3 - x)^2")

    for s in [1, 3]:
        a = tj.var(np.ones(s))
        b = tj.var(5)

        c = tj.mul(a, b)
        c = tj.add(c, 3)
        c = tj.sub(c, a)
        c = tj.mul(c, c)

        LOGGER.info("Initial a: %s" % a)

        for i in range(100):
            g = tj.gradients(c, [a])

            a.update(a.v - g[0] * 1e-2)

        LOGGER.info("Final a: %s" % a.v)

        assert _true(
            abs(a.v) < 1.0), "A should be smaller than 1 but is %s" % a
