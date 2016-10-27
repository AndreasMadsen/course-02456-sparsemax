import sparsemax
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops


###########################################################################
# Implementing sparsemax as OP in tensorflow
###########################################################################

# See https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342



###################################################################################
# Andreas sparsemax_forward function
###################################################################################

def forward(z, q):
    """Calculates the sparsemax loss function
    this will process a 2d-array $z$, where axis 1 (each row) is assumed to be
    the the z-vector. q is a binary matrix of same shape, containing the labels
    """

    # Calculate q^T * z
    z_k = np.sum(q * z, axis=1)

    # calculate sum over S(z)
    p = sparsemax.forward(z)
    s = p > 0
    # z_i^2 - tau(z)^2 = p_i (2 * z_i - p_i) for i \in S(z)
    S_sum = np.sum(s * p * (2 * z - p), axis=1)

    # because q is binary, sum([q_1^2, q_2^2, ...]) is just sum(q)
    q_norm = np.sum(q, axis=1)

    return -z_k + 0.5 * S_sum + 0.5 * q_norm

def grad(z, q):
    return -q + sparsemax.forward(z)

def _grad(op, grad):
	Z = op.inputs[0]
	q = op.inputs[1]
	result = -q + sparsemax.forward(Z.eval()) 
	return [result, None]

###################################################################################
# Define new op and gradient in Python
###################################################################################


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    
	# Need to generate a unique name to avoid duplicates:
	rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

	tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
	g = tf.get_default_graph()
	with g.gradient_override_map({"PyFunc": rnd_name}):
		result = tf.py_func(func, inp, Tout, stateful=stateful, name=name)
		return result


def sparsemax_forward(Z, q, name=None):
    
	with ops.op_scope([Z, q], name, "SparseMaxGrad") as name:

		# py_func takes a list of tensors and a function that takes np arrays as inputs
		# and returns np arrays as outputs
		forward_pass = py_func(forward,
							[Z, q],
							[tf.float64],
							name=name,
							grad=_grad)  # <-- here's the call to the gradient
		return forward_pass[0]

###################################################################################
# Run session and compare with numpy implementation
###################################################################################

with tf.Session() as sess:
	# Numpy implementation
	print("Numpy implementation: ")
	Z_np = np.array([[0.9,0.05,0.05],[0.1,0.3,0.5]])
	q_np = np.array([[1.0,0.0,0.0],[0.0,0.0,1.0]])
	print(forward(Z_np, q_np))

	print("Gradient: ")
	print(grad(Z_np, q_np))


	# Tensorflow OP
	Z = tf.constant([[0.9,0.05,0.05],[0.1,0.3,0.5]])
	q = tf.constant([[1.0,0.0,0.0],[0.0,0.0,1.0]])

	y = sparsemax_forward(Z, q)

	tf.initialize_all_variables().run()

	print("Tensorflow implementation: ")
	print(y.eval())
	print(tf.gradients(y, [Z,q])[0].eval())

