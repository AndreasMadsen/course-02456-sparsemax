import numpy as np


def forward(z):
    """forward pass for sparsemax

    this will process a 2d-array $z$, where axis 1 (each row) is assumed to be
    the the z-vector.
    """

    # sort z
    z_sorted = np.sort(z, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = np.cumsum(z_sorted, axis=1)
    k = np.arange(1, z.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    # use argmax to get the index by row as .nonzero() doesn't
    # take an axis argument. np.argmax return the first index, but the last
    # index is required here, use np.flip to get the last index and
    # `z.shape[axis]` to compensate for np.flip afterwards.
    k_z = z.shape[1] - np.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(z)
    tau_sum = z_cumsum[np.arange(0, z.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    # calculate p
    return np.maximum(0, z - tau_z)


def jacobian(z):
    """jacobian for sparsemax

    this will process a 2d-array $z$, where axis 1 (each row) is assumed to be
    the the z-vector.
    """

    # Construct S(z)
    # Possibly this could be reduced to just calculating k(z)
    p = forward(z)
    s = p > 0
    s_float = s.astype('float64')

    # row-wise outer product
    # http://stackoverflow.com/questions/31573856/theano-row-wise-outer-product-between-two-matrices
    jacobian = s_float[:, :, np.newaxis] * s_float[:, np.newaxis, :]
    jacobian /= - np.sum(s, axis=1)[:, np.newaxis, np.newaxis]

    # add delta_ij
    obs, index = s.nonzero()
    jacobian[obs, index, index] += 1

    return jacobian


def Rop(z, v):
    """Jacobian vector product (Rop) for sparsemax

    This calculates [J(z_i) * v_i, ...]. `z` is a 2d-array, where axis 1
    (each row) is assumed to be the the z-vector. `v` is a matrix where
    axis 1 (each row) is assumed to be the `v-vector`.
    """

    # Construct S(z)
    p = forward(z)
    s = p > 0

    # Calculate \hat{v}, which will be a vector (scalar for each z)
    v_hat = np.sum(v * s, axis=1) / np.sum(s, axis=1)

    # Calculates J(z) * v
    return s * (v - v_hat[:, np.newaxis])
