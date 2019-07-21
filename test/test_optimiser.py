"""optimiser test module."""
import tensorjo as tj
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)


def test_gd_on_linear_regression():
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

    opt = tj.opt.gd(err)
    opt.rounds = 1200

    opt.minimise([a, b])

    LOGGER.info("after training: coefficient %s -- bias: %s -- mse: %s" %
                (a, b, err.output()))


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


def test_gd_on_logistic_regression():
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

    opt = tj.opt.gd(err)
    opt.dt = 1e-0
    opt.rounds = 5000

    opt.minimise([a, b])

    LOGGER.info("after training: coefficient %s -- bias: %s -- mse: %s" %
                (a, b, err.output()))

    LOGGER.info("perfect would be 4 and -1")
    LOGGER.info("predictions %s", np.round(o.output()))
    LOGGER.info("observations %s", np.round(y))
