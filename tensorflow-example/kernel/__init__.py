
import os.path as path

import tensorflow as tf
from tensorflow.python.framework import ops, common_shapes

_thisdir = path.dirname(path.realpath(__file__))
_square_module = tf.load_op_library(path.join(_thisdir, 'square.so'))

square = _square_module.custom_square

# documentation says:
# @tf.RegisterShape("CustomSquare")(common_shapes.call_cpp_shape_fn)
# but that is not valid, syntax. From tensorflow source code it looks to be:
ops.RegisterShape("CustomSquare")(common_shapes.call_cpp_shape_fn)


@ops.RegisterGradient("CustomSquare")
def _zero_out_grad(op, grad):
    """The gradients for the CustomSquare op.

    Args:
    op: The `CustomSquare` `Operation` that we are differentiating, which we
      can use to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `CustomSquare` op.

    Returns:
    Gradients with respect to the input of `CustomSquare`.
    """
    # note op.inputs[0] is not yet evaluated, it is just the subgraph
    # representing the input. Same goes for the `op.outputs[i]`` and the
    # chain gradient `grad`.
    print('> debug: python grad')
    source = op.inputs[0]
    return [grad * (2 * source)]  # List of one Tensor, since we have one input
