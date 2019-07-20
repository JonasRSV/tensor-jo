"""Tensor module."""
import tensorjo as tj
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

invalid_tensors = [
    "cookie",
    bytes("cookie", encoding="utf8"), "", [], None, lambda x: x, 'a',
    [1, [1, 1], 1], [1, "cookie", 1]
]

valid_tensors = [
    0, -1, 1, 1.1, .1, 12312313123, [1, 2, 0, 1, 1], [1, 1.1, 2.0, 100],
    [[1.0, 1, 1, 1.0], [4, 1, 0, 1]],
    tj.tensor(np.array(500)),
    tj.tensor(5)
]


def test_tensor_initialization():
    """Check that only the correct types can be initialized as tensors."""
    LOGGER.info("Testing invalid tensor initializations.")
    for t in invalid_tensors:
        err = None
        tensor = None
        try:
            tensor = tj.tensor(t)
        except ValueError as e:
            err = e

        assert isinstance(err, ValueError),\
            "%s is supposed to be invalid -- %s : %s" % (t, tensor, err)

    LOGGER.info("Testing valid tensor initializations.")
    for t in valid_tensors:
        err = None
        try:
            tj.tensor(t)
        except Exception as e:
            err = e

        assert err is None,\
            "%s is supposed to be a valid tensor -- %s" % (t, err)


if __name__ == "__main__":
    test_tensor_initialization()
