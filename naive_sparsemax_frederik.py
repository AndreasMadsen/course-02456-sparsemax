import numpy as np

def sparsemax(z_array):
    """
    Computes the sparsemax of an independent

    Arguments
    ---------
    z_array : 2-d numpy array with each row representing a
        z_vector for a sample

    Returns
    --------
    a 2d-array of sparse probabilities
    """
    # step 1: sort
    sorted_z = np.sort(z_array, axis=1)[:, ::-1]

    p_array = np.empty(z_array.shape)
    for row in range(0, z_array.shape[0]):
        max_k = np.NaN

        # step 2: find argmax_k
        for k in range(1, z_array.shape[1]):
            if 1 + k * sorted_z[row, k - 1] > np.sum(sorted_z[row, :k]):
                max_k = k
            else:
                break

        # step 3: define tau
        tau_z = (np.sum(sorted_z[row, :max_k]) - 1) / max_k

        # step 4: set sparse probability vector
        p_array[row,:] = np.maximum(z_array[row, :] - tau_z, 0)

    #return array
    return p_array

def jacobian_sparsemax(p_vector):
    """
    Computes the jacobian of a sparsemax probability vector
    """

    # compute vector with i=1 if i in S(z)
    s = p_vector > 0
    jacobian = np.diag(s)  - np.outer(s, s.T) / np.sum(s)

    return jacobian

def jacobian_sparsemax_times_vector(p_vector, v_vector):
    s = p_vector > 0
    v_hat = np.sum(v_vector[s]) / (np.sum(s))

    return np.multiply(s, v_vector - v_hat)


oned_test = np.array([[5, 3, 0]])
twod_test = np.array([[5, 3, 0], [2, 2.7, 0]])
print(sparsemax(oned_test))
print(sparsemax(twod_test))

for i, row in enumerate(sparsemax(twod_test)):
    print(i)
    print(twod_test[i, :])
    print(row)
    print(jacobian_sparsemax(row))
    print(jacobian_sparsemax_times_vector(row, twod_test[i, :]))
    print(np.dot(jacobian_sparsemax(row), twod_test[i, :]))