"""Integration test module."""
import tensorjo as tj
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)


def test_linear_regression():
    """Test making simple 1d linear regression and train it."""
    x = np.arange(0, 10)

    # Co-domain
    y = x + 5

    # Variables
    a = tj.var(np.random.rand())
    b = tj.var(np.random.rand())

    err = tj.mse(y, a * x + b)

    LOGGER.info("Optimizing a simple 2 linear regression.")
    LOGGER.info(" X domain is [0, 1 .. 10]")
    LOGGER.info(" Y domain is [5, 6, .. 15]")

    LOGGER.info("before training: coefficient %s -- bias: %s -- mse: %s" %
                (a, b, err.output()))

    for i in range(1200):
        g = tj.gradients(err, [a, b])

        a.update(a.v - np.mean(g[0]) * 1e-2)
        b.update(b.v - np.mean(g[1]) * 1e-2)

    LOGGER.info("after training: coefficient %s -- bias: %s -- mse: %s" %
                (a, b, err.output()))


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


def test_logistic_regression():
    """Test making simple 1d linear regression and train it."""
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

    for i in range(5000):
        g = tj.gradients(err, [a, b])

        a.update(a.v - np.mean(g[0]) * 1e-0)
        b.update(b.v - np.mean(g[1]) * 1e-0)

    LOGGER.info("after training: coefficient %s -- bias: %s -- mse: %s" %
                (a, b, err.output()))

    LOGGER.info("perfect would be 4 and -1")
    LOGGER.info("predictions %s", np.round(o.output()))
    LOGGER.info("observations %s", np.round(y))
