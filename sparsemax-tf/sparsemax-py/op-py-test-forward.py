import sparsemax
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

###########################################################################
# OP for only the forward pass for sparsemax
###########################################################################


def forward(z, q):
    """Calculates the sparsemax loss function
    this will process a 2d-array $z$, where axis 1 (each row) is assumed to be
    the the z-vector. q is a binary matrix of same shape, containing the labels
    """

    # Calculate q^T * z
    z_k = np.sum(q * z, axis=1)
    print("z_k")
    print(z_k)

    # calculate sum over S(z)
    p = sparsemax.forward(z)
    s = p > 0
    # z_i^2 - tau(z)^2 = p_i (2 * z_i - p_i) for i \in S(z)
    S_sum = np.sum(s * p * (2 * z - p), axis=1)
    print(S_sum)
    # because q is binary, sum([q_1^2, q_2^2, ...]) is just sum(q)
    q_norm = np.sum(q, axis=1)
    print(q_norm)
    print("result")
    #result = np.array([-z_k + 0.5 * S_sum + 0.5 * ], dtype = float32)
    #result1 = np.square(z)
    #print(type(result[0]))
    #print(type(result1[0][0]))
    return np.array([-z_k + 0.5 * S_sum + 0.5 * q_norm])
    #return np.square(z)


with tf.Session() as sess:

	#Z = tf.constant([[0.9,0.05,0.05],[0.1,0.3,0.5]], dtype = tf.float32)
	#q = tf.constant([[1.0,0.0,0.0],[0.0,0.0,1.0]])


	Z = tf.constant([[1.]])
	q = tf.constant([[1.]])

	y = tf.py_func(forward, [Z, q], [tf.float64])
	
	tf.initialize_all_variables().run()

	print(y[0].eval())