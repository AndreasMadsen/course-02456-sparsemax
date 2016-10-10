import numpy as np

def sparsemax(z_array, return_tau=False):
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
    tau_vector = np.empty((z_array.shape[0], 1))
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
        tau_vector[row] = tau_z
        # step 4: set sparse probability vector
        p_array[row,:] = np.maximum(z_array[row, :] - tau_z, 0)

    #return array
    if return_tau:
        return [p_array, tau_vector]
    else:
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

def loss_sparsemax(z_vector, p_vector, target_label, tau_threshold):
    s = p_vector > 0
    return -z_vector[target_label] + 0.5 * np.sum(np.power(z_vector[s], 2) - tau_threshold ** 2 ) + 0.5

oned_test = np.array([[5, 3, 0]])
twod_test = np.array([[5, 3, 0], [2, 2.7, 0]])
twod_targets = np.array([0, 1])

print("trying to evaluate sparsemax")
print(sparsemax(oned_test))
print(sparsemax(twod_test))

print("\nTesting jacobian")
for i, row in enumerate(sparsemax(twod_test)):
    print("\ntesting row {0}".format(i))
    print("z: ", twod_test[i, :])
    print("sparse p: ", row)
    print("jacobian: ", jacobian_sparsemax(row))
    print("jacobian * z (indirectly): ", jacobian_sparsemax_times_vector(row, twod_test[i, :]))
    print("jacobian * z (directly): ", np.dot(jacobian_sparsemax(row), twod_test[i, :]))

[p_array, taus] = sparsemax(twod_test, return_tau=True)
print("\ntesting loss")
for i, tau in enumerate(taus):
    print("z:", twod_test[i, :])
    print("predicted: ", p_array[i, :])
    print("target: ", twod_targets[i])
    print("loss: ", loss_sparsemax(twod_test[i, :], p_array[i, :], twod_targets[i], tau))