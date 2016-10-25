import _test
from nose.tools import *

import tensorflow as tf
import numpy as np

import kernel

var = tf.placeholder(tf.int32, name='x')
data = np.asarray([[1, 2], [3, 4]], dtype='int32')
square = kernel.square(var)


def test_square_operator():
    """test square operator output"""

    with tf.Session():
        np.testing.assert_array_equal(
            square.eval({var: data}),
            np.asarray([[1, 4], [9, 16]], dtype='int32')
        )


def test_shape():
    """test square shape function"""
    with tf.Session():
        print(kernel.square(data).get_shape())
        np.testing.assert_array_equal(
            kernel.square(data).get_shape(),
            (2, 2)
        )

        np.testing.assert_array_equal(
            tf.shape(square).eval({var: data}),
            [2, 2]
        )


def test_gradient():
    """test square gradient output"""
    var_grad = tf.gradients(square, [var])[0]

    with tf.Session():
        np.testing.assert_array_equal(
            var_grad.eval({var: data}),
            np.asarray([[2, 4], [6, 8]], dtype='int32')
        )
