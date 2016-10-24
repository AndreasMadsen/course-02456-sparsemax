
import os.path as path

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops, common_shapes

_thisdir = path.dirname(path.realpath(__file__))
_sparsemax_module = tf.load_op_library(
    path.join(_thisdir, 'sparsemax.so'))
_sparsemax_loss_module = tf.load_op_library(
    path.join(_thisdir, 'sparsemax_loss.so'))

sparsemax = _sparsemax_module.sparsemax
sparsemax_loss = _sparsemax_loss_module.sparsemax_loss

# documentation says:
# @tf.RegisterShape("CustomSquare")(common_shapes.call_cpp_shape_fn)
# but that is not valid, syntax. From tensorflow source code it looks to be:
ops.RegisterShape("Sparsemax")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("SparsemaxLoss")(common_shapes.call_cpp_shape_fn)


@ops.RegisterGradient("Sparsemax")
def _sparsemax_grad(op, grad):
    """The gradients for the Sparsemax op.

    Args:
    op: The `Sparsemax` operation that we are differentiating, which we
      can use to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `Sparsemax` op.

    Returns:
    Gradients with respect to the input of `Sparsemax`.
    """
    # Construct S(z)
    sparsemax = op.outputs[0]
    support = tf.cast(sparsemax > 0, sparsemax.dtype)

    # Calculate \hat{v}, which will be a vector (scalar for each z)
    v_hat = tf.reduce_sum(tf.mul(grad, support), 1) / tf.reduce_sum(support, 1)

    # Calculates J(z) * v
    return [support * (grad - v_hat[:, np.newaxis])]
