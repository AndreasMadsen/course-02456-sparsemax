import _test
from nose.tools import *

import operator
import tensorflow as tf
import numpy as np

import sparsemax_tf_ops as ops


def sparsemax(z):
    logits = tf.placeholder(tf.float64, name='z')
    sparsemax = ops.sparsemax_op(logits)

    with tf.Session() as sess:
        return sparsemax.eval({logits: z})


def sparsemax_loss(z, q):
    logits = tf.placeholder(tf.float64, name='z')
    labels = tf.placeholder(tf.float64, name='q')
    sparsemax = ops.sparsemax_op(logits)
    loss = ops.sparsemax_loss_op(logits, sparsemax, labels)

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


def test_Rop_estimated():
    """check sparsemax-loss Rop, aginst estimated Rop"""
    z = np.random.uniform(low=-3, high=3, size=(100, 10))
    w = np.random.normal(1)
    q = np.zeros((100, 10))
    q[np.arange(0, 100), np.random.randint(0, 10, size=100)] = 1

    logits = tf.placeholder(tf.float64, name='z')
    labels = tf.constant(q, name='q')
    weight = tf.constant(w, name='w', dtype=tf.float64)

    sparsemax = ops.sparsemax_op(logits)
    loss = ops.sparsemax_loss_op(logits, sparsemax, labels)
    loss_transform = loss * weight

    with tf.Session() as sess:
        # https://www.tensorflow.org/versions/r0.8/api_docs/python/test.html
        analytical, numerical = tf.test.compute_gradient(
            logits, z.shape,
            loss_transform, (100, ),
            x_init_value=z, delta=1e-9
        )

        np.testing.assert_almost_equal(
            analytical,
            numerical,
            decimal=4
        )


def test_Rop_numpy():
    """check sparsemax-loss Rop, aginst numpy Rop"""
    z = np.random.uniform(low=-3, high=3, size=(100, 10))
    w = np.random.normal(size=(100, 1))
    q = np.zeros((100, 10))
    q[np.arange(0, 100), np.random.randint(0, 10, size=100)] = 1

    logits = tf.placeholder(tf.float64, name='z')
    labels = tf.constant(q, name='q')
    weights = tf.constant(w, name='w')

    sparsemax = ops.sparsemax_op(logits)
    loss = ops.sparsemax_loss_op(logits, sparsemax, labels)
    loss_transform = tf.expand_dims(loss, 1) * weights

    loss_transform_grad = tf.gradients(loss_transform, [logits])[0]

    with tf.Session() as sess:
        # chain rule
        grad = np.ones_like(w) * w

        np.testing.assert_array_equal(
            loss_transform_grad.eval({logits: z}),
            grad * (-q + sparsemax.eval({logits: z}))
        )
