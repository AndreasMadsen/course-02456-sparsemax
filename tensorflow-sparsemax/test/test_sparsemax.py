import _test
from nose.tools import *

import tensorflow as tf
import numpy as np

import kernel


def sparsemax(z):
    logits = tf.placeholder(tf.float64, name='z')
    sparsemax = kernel.sparsemax(logits)

    with tf.Session() as sess:
        return sparsemax.eval({logits: z})


def test_sparsemax_of_zero():
    """check sparsemax proposition 1, part 1"""
    z = np.zeros((1, 10))

    np.testing.assert_array_equal(
        sparsemax(z),
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
        sparsemax((1/epsilon) * z),
        p_expected
    )


def test_constant_add():
    """check sparsemax proposition 2"""
    z = np.random.uniform(low=-3, high=3, size=(100, 10))
    c = np.random.uniform(low=-3, high=3, size=(100, 1))

    np.testing.assert_almost_equal(
        sparsemax(z + c),
        sparsemax(z)
    )


def test_permutation():
    """check sparsemax proposition 3"""
    z = np.random.uniform(low=-3, high=3, size=(100, 10))
    p = sparsemax(z)

    for i in range(100):
        per = np.random.permutation(10)

        np.testing.assert_array_equal(
            sparsemax(z[i, per].reshape(1, -1)),
            p[i, per].reshape(1, -1)
        )


def test_diffrence():
    """check sparsemax proposition 4"""
    z = np.random.uniform(low=-3, high=3, size=(100, 10))
    p = sparsemax(z)

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

    p = sparsemax(z)
    p0_expected = np.select([t < -1, t <= 1, t > 1], [0, (t + 1) / 2, 1])

    np.testing.assert_almost_equal(p[:, 0], p0_expected)
    np.testing.assert_almost_equal(p[:, 1], 1 - p0_expected)


def test_Rop_estimated():
    """check sparsemax Rop, aginst estimated Rop"""
    z = np.random.uniform(low=-3, high=3, size=(100, 10))
    w = np.random.normal(size=(10, 10))

    logits = tf.placeholder(tf.float64, name='z')
    weights = tf.constant(w, name='w')
    sparsemax = kernel.sparsemax(logits)
    sparsemax_transform = tf.matmul(sparsemax, weights)

    with tf.Session() as sess:
        # https://www.tensorflow.org/versions/r0.8/api_docs/python/test.html
        analytical, numerical = tf.test.compute_gradient(
            logits, z.shape,
            sparsemax_transform, z.shape,
            x_init_value=z, delta=1e-9
        )

        np.testing.assert_almost_equal(
            analytical,
            numerical,
            decimal=4
        )


def test_Rop_numpy():
    """check sparsemax Rop, aginst numpy Rop"""
    z = np.random.uniform(low=-3, high=3, size=(100, 10))
    w = np.random.normal(size=(10, 10))

    logits = tf.placeholder(tf.float64, name='z')
    weights = tf.constant(w, name='w')
    sparsemax = kernel.sparsemax(logits)
    # tensorflow uses the chainrule forward (left to right), meaning that:
    #   $dS(z)*w/d(z) = dS(z)*w/dS(z) * dS(z)/dz = Rop(S)(z, Rop(*)(w, 1))$
    # Thus to test the Rop for sparsemax correctly a weight matrix is
    # multiplied. This causes the grad (v) in the Rop to be $dS(z)*w/dS(z)$.
    sparsemax_transform = tf.matmul(sparsemax, weights)
    sparsemax_transform_grad = tf.gradients(sparsemax_transform, [logits])[0]

    with tf.Session() as sess:
        # chain rule
        grad = np.dot(np.ones_like(z), w.T)

        # Construct S(z)
        properbility = sparsemax.eval({logits: z})
        support = properbility > 0

        # Calculate \hat{v}, which will be a vector (scalar for each z)
        v_hat = np.sum(grad * support, axis=1) / np.sum(support, axis=1)

        # Calculates J(z) * v
        numpy_grad = support * (grad - v_hat[:, np.newaxis])

        np.testing.assert_almost_equal(
            sparsemax_transform_grad.eval({logits: z}),
            numpy_grad
        )
