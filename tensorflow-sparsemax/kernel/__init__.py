
import os.path as path

import tensorflow as tf
from tensorflow.python.framework import ops, common_shapes

_thisdir = path.dirname(path.realpath(__file__))
_sparsemax_module = tf.load_op_library(path.join(_thisdir, 'sparsemax.so'))

sparsemax = _sparsemax_module.sparsemax

# documentation says:
# @tf.RegisterShape("CustomSquare")(common_shapes.call_cpp_shape_fn)
# but that is not valid, syntax. From tensorflow source code it looks to be:
ops.RegisterShape("Sparsemax")(common_shapes.call_cpp_shape_fn)
