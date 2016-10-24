import _test
from nose.tools import *

import operator
import tensorflow as tf
import numpy as np

import kernel


def sparsemax(z):
    logits = tf.placeholder(tf.float64, name='z')
    sparsemax = kernel.sparsemax(logits)

    with tf.Session() as sess:
        return sparsemax.eval({logits: z})


def sparsemax_loss(z, q):
    logits = tf.placeholder(tf.float64, name='z')
    labels = tf.placeholder(tf.float64, name='q')
    sparsemax = kernel.sparsemax(logits)
    loss = kernel.sparsemax_loss(logits, sparsemax, labels)

    with tf.Session() as sess:
        return loss.eval({logits: z, labels: q})


def test_constant_add():
    """check sparsemax-loss proposition 3"""
    z = np.random.uniform(low=-3, high=3, size=(100, 10))
    c = np.random.uniform(low=-3, high=3, size=(100, 1))
    q = np.zeros((100, 10))
    q[np.arange(0, 100), np.random.randint(0, 10, size=100)] = 1

    np.testing.assert_almost_equal(
        sparsemax_loss(z, q),
        sparsemax_loss(z + c, q)
    )


def test_positive():
    """check sparsemax-loss proposition 4"""
    z = np.random.uniform(low=-3, high=3, size=(100, 10))
    q = np.zeros((100, 10))
    q[np.arange(0, 100), np.random.randint(0, 10, size=100)] = 1

    loss = sparsemax_loss(z, q)
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
        sparsemax_loss(z, q),
        0
    )

    np.testing.assert_almost_equal(
        sparsemax(z),
        q
    )
