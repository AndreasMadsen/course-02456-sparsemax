import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import tensorflow_python.sparsemax
import tensorflow_python.sparsemax_loss


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        result = tf.py_func(func, inp, Tout, stateful=stateful, name=name)
        return result


def grad_sparsemax(op, grad):
    spm = op.outputs[0]
    support = tf.cast(spm > 0, spm.dtype)

    # Calculate \hat{v}, which will be a vector (scalar for each z)
    v_hat = tf.reduce_sum(tf.mul(grad, support), 1) / tf.reduce_sum(support, 1)

    # Calculates J(z) * v
    return [support * (grad - v_hat[:, np.newaxis])]


def grad_sparsemax_loss(op, grad):
    spm = op.inputs[1]
    labels = op.inputs[2]
    result = tf.transpose(grad * tf.transpose(-labels + spm))
    return [result, None, None]


def sparsemax_op(Z, name=None):
    with tf.name_scope(name, "SparseMaxGrad", [Z]) as name:
        # py_func takes a list of tensors and a
        # function that takes np arrays as inputs
        # and returns np arrays as outputs
        sparsemax_forward = py_func(
            tensorflow_python.sparsemax.forward,
            [Z],
            [tf.float64],
            name=name,
            grad=grad_sparsemax)  # <-- here's the call to the gradient
    return sparsemax_forward[0]


def sparsemax_loss_op(Z, sparsemax, q, name=None):
    with tf.name_scope(name, "SparseMaxLossGrad", [Z, sparsemax, q]) as name:
        # py_func takes a list of tensors and a
        # function that takes np arrays as inputs
        # and returns np arrays as outputs
        sparsemax_forward_loss = py_func(
            tensorflow_python.sparsemax_loss.forward_loss,
            [Z, sparsemax, q],
            [tf.float64],
            name=name,
            grad=grad_sparsemax_loss)  # <-- here's the call to the gradient

    return sparsemax_forward_loss[0]
