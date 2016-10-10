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

oned_test = np.array([[5, 3, 0]])
twod_test = np.array([[5, 3, 0], [2, 2.7, 0]])
print(sparsemax(oned_test))
print(sparsemax(twod_test))