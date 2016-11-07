import numpy as np
import tensorflow_python.sparsemax


def forward_loss(z, spm, q):
    """Calculates the sparsemax loss function
    this will process a 2d-array $z$, where axis 1 (each row) is assumed to be
    the the z-vector. q is a binary matrix of same shape, containing the labels
    """

    # Calculate q^T * z
    z_k = np.sum(q * z, axis=1)

    # calculate sum over S(z)
    p = spm
    s = p > 0
    # z_i^2 - tau(z)^2 = p_i (2 * z_i - p_i) for i \in S(z)
    S_sum = np.sum(s * p * (2 * z - p), axis=1)

    # because q is binary, sum([q_1^2, q_2^2, ...]) is just sum(q)
    q_norm = np.sum(q, axis=1)

    return -z_k + 0.5 * S_sum + 0.5 * q_norm


def grad(z, q):
    return -q + sparsemax.forward(z)
