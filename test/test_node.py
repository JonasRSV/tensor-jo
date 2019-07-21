"""Tensor module."""
import tensorjo as tj
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

ok_numerical_error = 1e-3


def _true(item):
    try:
        return all(np.array(item).reshape(-1))
    except Exception as e:
        return item


def test_primitive():
    """Check core functionality of primitive nodes."""
    pi = np.array(np.random.rand())
    p0i = np.random.rand(5)
    p1i = np.random.rand(5, 1)
    p2i = np.random.rand(5, 5)

    p = tj.node.primitive(pi, name="p")
    p0 = tj.node.primitive(p0i, name="p0")
    p1 = tj.node.primitive(p1i, name="p1")
    p2 = tj.node.primitive(p2i, name="p2")

    a = zip([pi, p0i, p1i, p2i], [p, p0, p1, p2], ["p", "p0", "p1", "p2"])

    LOGGER.info("Testing constructor, name, output and shape of primitive.")
    for inner, prim, name in a:
        assert _true(abs(inner - prim.output()) < ok_numerical_error),\
            "%s should be %s" % (prim.v, inner)
        assert inner.shape == prim.shape(), "%s should be %s"\
            % (prim.shape(), inner.shape)

        assert prim.name == name, "%s should be %s" % (prim.name, name)


monoids = {
    "add": {
        "m1": [
            np.array(np.random.rand()),
            np.random.rand(5),
            np.random.rand(5, 1),
            np.random.rand(5, 5),
            tj.node.primitive(
                np.array(np.random.rand()), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 1), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 5), name=tj.naming.get_tensor_name())
        ],
        "m2": [
            tj.node.primitive(
                np.array(np.random.rand()), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 1), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 5), name=tj.naming.get_tensor_name()),
            np.array(np.random.rand()),
            np.random.rand(5),
            np.random.rand(5, 1),
            np.random.rand(5, 5)
        ],
        "correct":
        lambda x, y: x + y
    },
    "sub": {
        "m1": [
            np.array(np.random.rand()),
            np.random.rand(5),
            np.random.rand(5, 1),
            np.random.rand(5, 5),
            tj.node.primitive(
                np.array(np.random.rand()), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 1), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 5), name=tj.naming.get_tensor_name())
        ],
        "m2": [
            tj.node.primitive(
                np.array(np.random.rand()), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 1), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 5), name=tj.naming.get_tensor_name()),
            np.array(np.random.rand()),
            np.random.rand(5),
            np.random.rand(5, 1),
            np.random.rand(5, 5)
        ],
        "correct":
        lambda x, y: x - y
    },
    "mul": {
        "m1": [
            np.array(np.random.rand()),
            np.random.rand(5),
            np.random.rand(5, 1),
            np.random.rand(5, 5),
            tj.node.primitive(
                np.array(np.random.rand()), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 1), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 5), name=tj.naming.get_tensor_name())
        ],
        "m2": [
            tj.node.primitive(
                np.array(np.random.rand()), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 1), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 5), name=tj.naming.get_tensor_name()),
            np.array(np.random.rand()),
            np.random.rand(5),
            np.random.rand(5, 1),
            np.random.rand(5, 5)
        ],
        "correct":
        lambda x, y: x * y
    },
    "div": {
        "m1": [
            np.array(np.random.rand()),
            np.random.rand(5),
            np.random.rand(5, 1),
            np.random.rand(5, 5),
            tj.node.primitive(
                np.array(np.random.rand()), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 1), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 5), name=tj.naming.get_tensor_name())
        ],
        "m2": [
            tj.node.primitive(
                np.array(np.random.rand()), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 1), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 5), name=tj.naming.get_tensor_name()),
            np.array(np.random.rand()),
            np.random.rand(5),
            np.random.rand(5, 1),
            np.random.rand(5, 5)
        ],
        "correct":
        lambda x, y: x / (y + 1e-15)
    },
    "mse": {
        "m1": [
            np.random.rand(5, 1),
            np.random.rand(5, 5),
            tj.node.primitive(
                np.random.rand(5, 1), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 5), name=tj.naming.get_tensor_name())
        ],
        "m2": [
            tj.node.primitive(
                np.random.rand(5, 1), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 5), name=tj.naming.get_tensor_name()),
            np.random.rand(5, 1),
            np.random.rand(5, 5)
        ],
        "correct":
        lambda x, y: np.mean(np.square(x - y))
    },
}


def test_monoid():
    """Check core functionality of monoid nodes."""
    for op, context in monoids.items():
        LOGGER.info("Testing %s monoid" % op)

        for m1, m2 in zip(context["m1"], context["m2"]):
            o = eval("tj.%s(m1, m2)" % op).output()

            if hasattr(m1, "v"):
                m1 = m1.v

            if hasattr(m2, "v"):
                m2 = m2.v

            assert _true(abs(o - context["correct"](m1, m2))
                         < ok_numerical_error),\
                "%s (%s) %s is not %s" % (m1, op, m2, o)

            # TODO: Add checks for shape and stuff.


functors = {
    "sigmoid": {
        "m": [
            np.array(np.random.rand()),
            np.random.rand(5),
            np.random.rand(5, 1),
            np.random.rand(5, 5),
            tj.node.primitive(
                np.array(np.random.rand()), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 1), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 5), name=tj.naming.get_tensor_name())
        ],
        "correct":
        lambda x: np.exp(x) / (1 + np.exp(x))
    },
    "sin": {
        "m": [
            np.array(np.random.rand()),
            np.random.rand(5),
            np.random.rand(5, 1),
            np.random.rand(5, 5),
            tj.node.primitive(
                np.array(np.random.rand()), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 1), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 5), name=tj.naming.get_tensor_name())
        ],
        "correct":
        np.sin
    },
    "cos": {
        "m": [
            np.array(np.random.rand()),
            np.random.rand(5),
            np.random.rand(5, 1),
            np.random.rand(5, 5),
            tj.node.primitive(
                np.array(np.random.rand()), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 1), name=tj.naming.get_tensor_name()),
            tj.node.primitive(
                np.random.rand(5, 5), name=tj.naming.get_tensor_name())
        ],
        "correct":
        np.cos
    }
}


def test_functor():
    """Check core functionality of functor nodes."""
    for op, context in functors.items():
        LOGGER.info("Testing %s functor" % op)

        for m in context["m"]:
            o = eval("tj.%s(m)" % op).output()

            if hasattr(m, "v"):
                m = m.v

            assert _true(abs(o - context["correct"](m))
                         < ok_numerical_error),\
                "%s(%s) is not %s" % (op, m, o)

            assert o.shape == context["correct"](m).shape,\
                "Functor shape missmatch %s(%s) should %s but got %s"\
                % (op, context["correct"].shape, o.shape)
