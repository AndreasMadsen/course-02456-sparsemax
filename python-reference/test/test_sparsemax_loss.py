import _test
from nose.tools import *

import warnings
import operator
import numpy as np
import numdifftools

import sparsemax
import sparsemax_loss


def test_constant_add():
    """check sparsemax-loss proposition 3"""
    z = np.random.uniform(low=-3, high=3, size=(100, 10))
    c = np.random.uniform(low=-3, high=3, size=(100, 1))
    q = np.zeros((100, 10))
    q[np.arange(0, 100), np.random.randint(0, 10, size=100)] = 1

    np.testing.assert_almost_equal(
        sparsemax_loss.forward(z, q),
        sparsemax_loss.forward(z + c, q)
    )


def test_positive():
    """check sparsemax-loss proposition 4"""
    z = np.random.uniform(low=-3, high=3, size=(100, 10))
    q = np.zeros((100, 10))
    q[np.arange(0, 100), np.random.randint(0, 10, size=100)] = 1

    loss = sparsemax_loss.forward(z, q)
    np.testing.utils.assert_array_compare(
        operator.__ge__, loss, np.zeros_like(loss)
    )


def test_zero_loss():
    """check sparsemax-loss proposition 5"""
    # construct z and q, such that z_k >= 1 + max_{j!=k} z_k holds for
    # delta_0 = 1.
    z = np.random.uniform(low=-3, high=3, size=(100, 10))
    z[:, 0] = np.max(z, axis=1) + 1.05

    q = np.zeros((100, 10))
    q[:, 0] = 1

    np.testing.assert_almost_equal(
        sparsemax_loss.forward(z, q),
        0
    )

    np.testing.assert_almost_equal(
        sparsemax.forward(z),
        q
    )


def test_gradient():
    """check sparsemax-loss gradient, against approximation"""
    t = np.linspace(-2, 2, 100)
    z = np.vstack([
        t, np.zeros(100)
    ]).T
    q = np.zeros((100, 2))
    q[np.arange(0, 100), np.random.randint(0, 2, size=100)] = 1

    gradient_exact = sparsemax_loss.grad(z, q)

    for i, (z_i, q_i) in enumerate(zip(z, q)):
        gradient_approx = numdifftools.Gradient(
            lambda z_i: sparsemax_loss.forward(
                z_i[np.newaxis, :], q_i[np.newaxis, :]
            )[0]
        )

        with warnings.catch_warnings(record=True) as w:
            # numdifftools causes a warning, a better filter could be made
            np.testing.assert_almost_equal(
                gradient_exact[i, :],
                gradient_approx(z_i)
            )
