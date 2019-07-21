"""Tensor Module."""
import tensorjo as tj
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

ok_numerical_error = 1e-6

tensors = {
    "1d": {
        "m1": list(map(np.array, [1, -1, 0.1, 1.0, 10, 100, -100, 0])),
        "m2": list(map(np.array, [1, -1, 0.1, 100, 10, -10, 100, 0]))
    },
    "2d": {
        "m1":
        list(
            map(np.array, [
                np.ones((10, 10)),
                np.zeros((10, 10)),
                np.random.rand(10, 10)
            ])),
        "m2":
        list(
            map(np.array, [
                np.ones((10, 10)),
                np.zeros((10, 10)),
                np.random.rand(10, 10), 10, 100
            ]))
    },
    "invalid": {
        "m1":
        list(
            map(np.array,
                [np.ones((10, 10)),
                 np.ones((10, 11)),
                 np.random.rand(3, 4)])),
        "m2":
        list(
            map(np.array,
                [np.ones((11, 10)),
                 np.ones((11, 10)),
                 np.random.rand(4, 4)]))
    },
    "derivatives": {
        "m1":
        list(map(np.array,
                 [np.ones(5), np.zeros(5),
                  np.ones((5, 5)), 1, 2])),
        "m2":
        list(
            map(np.array,
                [np.ones(5) * 2,
                 np.ones(5),
                 np.ones((5, 5)) * 2, 5, 0]))
    },
    "invalid_mse": {
        "m1":
        list(
            map(np.array,
                [np.ones((10, 10)),
                 np.ones((10, 11)),
                 np.random.rand(3, 4)])),
        "m2":
        list(
            map(np.array,
                [np.ones((11, 10)),
                 np.ones((11, 10)),
                 np.random.rand(4, 4)]))
    },
    "derivative_mse": {
        "m1":
        list(
            map(np.array, [
                np.ones((10, 10)),
                np.zeros((10, 10)),
                np.ones((10, 1)),
                np.ones((20, 1)) * 40,
            ])),
        "m2":
        list(
            map(np.array, [
                np.ones((10, 10)),
                np.zeros((10, 10)),
                np.ones((10, 1)) * 10,
                np.ones((20, 3)) * 3
            ]))
    },
    "functors": {
        "m":
        list(
            map(np.array, [
                np.ones(1),
                np.ones((10, 10)),
                np.ones((10, 1)),
                np.ones((5, 2))
            ])),
    },
}


def _true(item):
    try:
        return all(np.array(item).reshape(-1))
    except Exception as e:
        return item


def test_addition():
    """Test the addition Op."""
    LOGGER.info("Testing valid addition ops.")

    for tensor in ["1d", "2d"]:
        _d = tensors[tensor]

        for m1, m2 in zip(_d["m1"], _d["m2"]):
            add_op = tj.ops.addition(m1, m2)
            add_op_res = add_op.forward(m1, m2)
            assert _true(add_op_res == (m1 + m2)),\
                "Forward add_op gave wrong result: %s + %s != %s"\
                % (m1, m2, add_op_res)

    LOGGER.info("Testing invalid addition ops.")

    _d = tensors["invalid"]

    for m1, m2 in zip(_d["m1"], _d["m2"]):

        exception = None
        try:
            tj.ops.addition(m1, m2)
        except ValueError as e:
            exception = e

        assert isinstance(exception, ValueError),\
            "An exception should have been thrown %s & %s cannot be added"\
            % (m1.shape, m2.shape)

    LOGGER.info("Testing derivative of addition ops.")

    _d = tensors["derivatives"]

    for m1, m2 in zip(_d["m1"], _d["m2"]):
        add_op = tj.ops.addition(m1, m2)

        first_deriv = add_op.backward_first()
        assert _true(first_deriv == np.ones_like(m1)),\
            "derivative of %s + %s with respect to the first is not %s" \
            % (m1, m2, first_deriv)

        second_deriv = add_op.backward_second()
        assert _true(second_deriv == np.ones_like(m2)),\
            "derivative of %s + %s with respect to the second is not %s" \
            % (m1, m2, second_deriv)

    LOGGER.info("Testing shape of addition ops.")

    for tensor in ["1d", "2d"]:
        _d = tensors[tensor]

        for m1, m2 in zip(_d["m1"], _d["m2"]):
            add_op = tj.ops.addition(m1, m2)
            add_op_shape = add_op.shape()
            assert _true(add_op_shape == (m1 + m2).shape),\
                "Shape add_op gave wrong result: %s + %s != %s"\
                % (m1.shape, m2.shape, add_op_shape)


def test_subtraction():
    """Test the subtraction Op."""
    LOGGER.info("Testing valid subtraction ops.")

    for tensor in ["1d", "2d"]:
        _d = tensors[tensor]

        for m1, m2 in zip(_d["m1"], _d["m2"]):
            sub_op = tj.ops.subtraction(m1, m2)
            sub_op_res = sub_op.forward(m1, m2)
            assert _true(sub_op_res == (m1 - m2)),\
                "Forward sub_op gave wrong result: %s - %s != %s"\
                % (m1, m2, sub_op_res)

    LOGGER.info("Testing invalid subtraction ops.")

    _d = tensors["invalid"]

    for m1, m2 in zip(_d["m1"], _d["m2"]):

        exception = None
        try:
            tj.ops.subtraction(m1, m2)
        except ValueError as e:
            exception = e

        assert isinstance(exception, ValueError),\
            "An exception should have been thrown %s & %s cannot be subbed"\
            % (m1.shape, m2.shape)

    LOGGER.info("Testing derivative of subtraction ops.")

    _d = tensors["derivatives"]

    for m1, m2 in zip(_d["m1"], _d["m2"]):
        sub_op = tj.ops.subtraction(m1, m2)
        first_deriv = sub_op.backward_first()
        assert _true(first_deriv == np.ones_like(m1)),\
            "derivative of %s - %s with respect to the first is not %s" \
            % (m1, m2, first_deriv)

        second_deriv = sub_op.backward_second()
        assert _true(second_deriv == -np.ones_like(m2)),\
            "derivative of %s - %s with respect to the second is not %s" \
            % (m1, m2, second_deriv)

    LOGGER.info("Testing shape of subtraction ops.")

    for tensor in ["1d", "2d"]:
        _d = tensors[tensor]

        for m1, m2 in zip(_d["m1"], _d["m2"]):
            sub_op = tj.ops.subtraction(m1, m2)
            sub_op_shape = sub_op.shape()
            assert _true(sub_op_shape == (m1 + m2).shape),\
                "Shape sub_op gave wrong result: %s - %s != %s"\
                % (m1.shape, m2.shape, sub_op_shape)


def test_multiplication():
    """Test the multplication Op."""
    LOGGER.info("Testing valid multiplication ops.")
    for tensor in ["1d", "2d"]:
        _d = tensors[tensor]

        for m1, m2 in zip(_d["m1"], _d["m2"]):
            mult_op = tj.ops.multiplication(m1, m2)
            mult_op_res = mult_op.forward(m1, m2)
            assert _true(mult_op_res == (m1 * m2)),\
                "Forward mult_op gave wrong result: %s * %s != %s"\
                % (m1, m2, mult_op_res)

    LOGGER.info("Testing invalid multiplication ops.")

    _d = tensors["invalid"]

    for m1, m2 in zip(_d["m1"], _d["m2"]):

        exception = None
        try:
            tj.ops.multiplication(m1, m2)
        except ValueError as e:
            exception = e

        assert isinstance(exception, ValueError),\
            "An exception should have been thrown %s & %s cannot be mult"\
            % (m1.shape, m2.shape)

    LOGGER.info("Testing derivative of multiplication ops.")

    _d = tensors["derivatives"]

    for m1, m2 in zip(_d["m1"], _d["m2"]):
        mul_op = tj.ops.multiplication(m1, m2)
        first_deriv = mul_op.backward_first()
        assert _true(first_deriv == m2),\
            "derivative of %s * %s with respect to the first is not %s" \
            % (m1, m2, first_deriv)

        second_deriv = mul_op.backward_second()
        assert _true(second_deriv == m1),\
            "derivative of %s * %s with respect to the second is not %s"\
            % (m1, m2, second_deriv)

    LOGGER.info("Testing shape of multiplication ops.")

    for tensor in ["1d", "2d"]:
        _d = tensors[tensor]

        for m1, m2 in zip(_d["m1"], _d["m2"]):
            mul_op = tj.ops.multiplication(m1, m2)
            mul_op_shape = mul_op.shape()
            assert _true(mul_op_shape == (m1 + m2).shape),\
                "Shape mul_op gave wrong result: %s * %s != %s"\
                % (m1.shape, m2.shape, mul_op_shape)


def test_division():
    """Test the divison Op."""
    LOGGER.info("Testing valid division ops.")

    for tensor in ["1d", "2d"]:
        _d = tensors[tensor]

        for m1, m2 in zip(_d["m1"], _d["m2"]):
            div_op = tj.ops.division(m1, m2)
            div_op_res = div_op.forward(m1, m2)
            assert _true(abs(div_op_res -
                         (m1 / (m2 + tj.ops.division.tiny_number)))
                         < ok_numerical_error),\
                "Forward division_op gave wrong result: %s / %s != %s"\
                % (m1, m2, div_op_res)

    LOGGER.info("Testing invalid division ops.")

    _d = tensors["invalid"]

    for m1, m2 in zip(_d["m1"], _d["m2"]):

        exception = None
        try:
            tj.ops.division(m1, m2)
        except ValueError as e:
            exception = e

        assert isinstance(exception, ValueError),\
            "An exception should have been thrown %s & %s cannot be div"\
            % (m1.shape, m2.shape)

    LOGGER.info("Testing derivative of division ops.")

    _d = tensors["derivatives"]

    for m1, m2 in zip(_d["m1"], _d["m2"]):
        div_op = tj.ops.division(m1, m2)

        first_deriv = div_op.backward_first()
        assert _true(abs(first_deriv - (1 / (m2 + div_op.tiny_number)))
                     < ok_numerical_error),\
            "derivative of %s / %s with respect to the first is not %s" \
            % (m1, m2, first_deriv)

        second_deriv = div_op.backward_second()
        assert _true(abs(second_deriv - (-m1 / (m2 * m2 + div_op.tiny_number)))
                     < ok_numerical_error),\
            "derivative of %s / %s with respect to the second is not %s" \
            % (m1, m2, second_deriv)

    LOGGER.info("Testing shape of division ops.")

    for tensor in ["1d", "2d"]:
        _d = tensors[tensor]

        for m1, m2 in zip(_d["m1"], _d["m2"]):
            div_op = tj.ops.division(m1, m2)
            div_op_shape = div_op.shape()
            assert _true(div_op_shape == (m1 + m2).shape),\
                "Shape div_op gave wrong result: %s / %s != %s"\
                % (m1.shape, m2.shape, div_op_shape)


def test_mse():
    """Test the mse Op."""
    LOGGER.info("Testing valid mse ops.")

    for tensor in ["2d"]:
        _d = tensors[tensor]

        for m1, m2 in zip(_d["m1"], _d["m2"]):
            mse_op = tj.ops.mse(m1, m2)
            mse_op_res = mse_op.forward(m1, m2)
            assert _true(abs(mse_op_res - np.mean(np.square(m1 - m2)))
                         < ok_numerical_error),\
                "Forward mse gave wrong result: mse(%s, %s) != %s"\
                % (m1, m2, mse_op_res)

    LOGGER.info("Testing invalid mse ops.")

    _d = tensors["invalid_mse"]

    for m1, m2 in zip(_d["m1"], _d["m2"]):

        exception = None
        try:
            tj.ops.mse(m1, m2)
        except ValueError as e:
            exception = e

        assert isinstance(exception, ValueError),\
            "An exception should have been thrown %s & %s cannot be mse"\
            % (m1.shape, m2.shape)

    LOGGER.info("Testing derivative of mse ops.")

    _d = tensors["derivative_mse"]

    for m1, m2 in zip(_d["m1"], _d["m2"]):
        mse_op = tj.ops.mse(m1, m2)
        first_deriv = mse_op.backward_first()
        diff = m1 - m2
        assert _true(abs(first_deriv - (2 * diff))
                     < ok_numerical_error),\
            "derivative of mse(%s, %s) with respect to the first is not %s" \
            % (m1, m2, first_deriv)

        second_deriv = mse_op.backward_second()
        assert _true(abs(second_deriv - (-2 * diff))
                     < ok_numerical_error),\
            "derivative of mse(%s, %s) with respect to the second is not %s" \
            % (m1, m2, second_deriv)

    LOGGER.info("Testing shape of mse ops.")

    for tensor in ["2d"]:
        _d = tensors[tensor]

        for m1, m2 in zip(_d["m1"], _d["m2"]):
            mse_op = tj.ops.mse(m1, m2)
            mse_op_shape = mse_op.shape()
            assert _true(mse_op_shape ==
                         (np.mean(np.square(m1 - m2))).shape),\
                "Shape mse_op gave wrong result: mse(%s, %s) != %s"\
                % (m1.shape, m2.shape, mse_op_shape)


def test_functors():
    """Test the functors Op."""
    LOGGER.info("Testing valid functors.")

    functors = [("sigmoid", lambda x: np.exp(x) / (1 + np.exp(x))),
                ("sin", np.sin), ("cos", np.cos)]

    domain = tensors["functors"]["m"]

    for f, correct in functors:
        LOGGER.info("    Testing: %s" % f)
        for d in domain:
            f_op = eval("tj.ops.%s(d)" % f)
            f_op_res = f_op.forward(d)

            assert _true(abs(f_op_res - correct(d)) < ok_numerical_error),\
                "%s gave wrong result %s(%s) != %s"\
                % (f, f, f_op_res, correct(d))

    def sigmoid(x):
        return np.exp(x) / (1 + np.exp(x))

    functors = [("sigmoid", lambda x: sigmoid(x) * (1 - sigmoid(x))),
                ("sin", np.cos), ("cos", lambda x: -np.sin(x))]

    LOGGER.info("Testing derivative of functor ops.")
    for f, correct in functors:
        LOGGER.info("    Testing: %s" % f)
        for d in domain:
            f_op = eval("tj.ops.%s(d)" % f)
            f_op_res = f_op.backward_functor()

            assert _true(abs(f_op_res - correct(d)) < ok_numerical_error),\
                "%s gave wrong result %s(%s) != %s"\
                % (f, f, f_op_res, correct(d))

    LOGGER.info("Testing shape of functor ops.")
    functors = [("sigmoid", lambda x: sigmoid(x)), ("sin", np.sin), ("cos",
                                                                     np.cos)]

    for f, correct in functors:
        LOGGER.info("    Testing: %s" % f)
        for d in domain:
            f_op = eval("tj.ops.%s(d)" % f)
            res = f_op.forward(d)

            assert _true(res.shape == correct(d).shape),\
                "%s gave wrong shape %s(%s) != %s"\
                % (f, f, res.shape, correct(d).shape)
