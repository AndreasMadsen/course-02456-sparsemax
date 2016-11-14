import _test
from nose.tools import *

import warnings
import numpy as np
import numdifftools

import sparsemax


def test_sparsemax_of_zero():
    """check sparsemax proposition 1, part 1"""
    z = np.zeros((1, 10))

    np.testing.assert_array_equal(
        sparsemax.forward(z),
        np.ones_like(z) / z.size
    )


def test_sparsemax_of_inf():
    """check sparsemax proposition 1, part 2"""
    z = np.random.uniform(low=-3, high=3, size=(100, 10))

    # assume |A(z)| = 1, as z is continues random
    z_sort_arg = np.argsort(z, axis=1)[:, ::-1]
    z_sort = np.sort(z, axis=-1)[:, ::-1]
    gamma_z = z_sort[:, 0] - z_sort[:, 1]
    epsilon = (0.99 * gamma_z * 1).reshape(-1, 1)

    # construct the expected 1_A(z) array
    p_expected = np.zeros((100, 10))
    p_expected[np.arange(0, 100), z_sort_arg[:, 0]] = 1

    np.testing.assert_almost_equal(
        sparsemax.forward((1/epsilon) * z),
        p_expected
    )


def test_constant_add():
    """check sparsemax proposition 2"""
    z = np.random.uniform(low=-3, high=3, size=(100, 10))
    c = np.random.uniform(low=-3, high=3, size=(100, 1))

    np.testing.assert_almost_equal(
        sparsemax.forward(z + c),
        sparsemax.forward(z)
    )


def test_permutation():
    """check sparsemax proposition 3"""
    z = np.random.uniform(low=-3, high=3, size=(100, 10))
    p = sparsemax.forward(z)

    for i in range(100):
        per = np.random.permutation(10)

        np.testing.assert_array_equal(
            sparsemax.forward(z[i, per].reshape(1, -1)),
            p[i, per].reshape(1, -1)
        )


def test_diffrence():
    """check sparsemax proposition 4"""
    z = np.random.uniform(low=-3, high=3, size=(100, 10))
    p = sparsemax.forward(z)

    for val in range(0, 100):
        for i in range(0, 10):
            for j in range(0, 10):
                # check condition, the obesite pair will be checked anyway
                if z[val, i] > z[val, j]:
                    continue

                assert_true(
                    0 <= p[val, j] - p[val, i] <= z[val, j] - z[val, i] + 1e-9,
                    "0 <= %.10f <= %.10f" % (
                        p[val, j] - p[val, i], z[val, j] - z[val, i]
                    )
                )


def test_two_dimentional():
    """check two dimentation sparsemax case"""
    t = np.linspace(-2, 2, 100)
    z = np.vstack([
        t, np.zeros(100)
    ]).T

    p = sparsemax.forward(z)
    p0_expected = np.select([t < -1, t <= 1, t > 1], [0, (t + 1) / 2, 1])

    np.testing.assert_almost_equal(p[:, 0], p0_expected)
    np.testing.assert_almost_equal(p[:, 1], 1 - p0_expected)


def test_two_dimentional_jacobian():
    """check two dimentation sparsemax jacobian against approiximation"""
    t = np.linspace(-2, 2, 100)
    z = np.vstack([
        t, np.zeros(100)
    ]).T

    Jacobian_approx = numdifftools.Jacobian(
        lambda z_i: sparsemax.forward(z_i[np.newaxis, :])[0]
    )

    jacobian_exact = sparsemax.jacobian(z)

    for i, z_i in enumerate(z):
        with warnings.catch_warnings(record=True) as w:
            # numdifftools causes a warning, a better filter could be made
            np.testing.assert_almost_equal(
                jacobian_exact[i, :, :],
                Jacobian_approx(z_i)
            )


def test_Rop():
    """check sparsemax jacobian times vector (Rop), aginst exact jacobian"""
    z = np.random.uniform(low=-3, high=3, size=(100, 10))
    v = np.random.uniform(low=-3, high=3, size=(100, 10))

    jacobian_exact = sparsemax.jacobian(z)

    np.testing.assert_almost_equal(
        np.einsum('oij,oi->oj', jacobian_exact, v),  # J(z) * v for each obs.
        sparsemax.Rop(z, v)
    )


def test_sparsemax_on_examples():
    """ check sparsemax on a harcoded examples"""
    twod_input_vector = np.array([[5, 3, 0], [2, 2.7, 0]])
    value_expected = np.array([[1, 0, 0],
                              [((2 + 1) - 2.7) / 2,
                               ((2.7 + 1) - 2) / 2, 0]])

    output_value = sparsemax.forward(twod_input_vector)
    np.testing.assert_almost_equal(output_value, value_expected)
