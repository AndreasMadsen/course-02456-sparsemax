## naive implementation of sparsemax
## (made for learning purposes)
## test this file with "nosetests naive_sparsemax_frederik.py"

import numpy as np
from matplotlib import pyplot as plt
import unittest

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
        for k in range(1, z_array.shape[1] + 1):
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


class Tests(unittest.TestCase):
    def test_evaluate_function_numpy2darrays(self):
        # test that function evaluations gives the expected output
        one_row_input = np.array([[5, 3, 0]])
        res_one = sparsemax(one_row_input)
        for pair in zip(res_one.ravel(), [1, 0, 0]):
            self.assertEqual(pair[0], pair[1]) 
        
        # test that it also works in the case of multiple input rows
        two_rows_input = np.array([[5, 3, 0], [2, 2.7, 0]])
        res_two = sparsemax(two_rows_input);
        print(res_two)
        expected = [[1, 0, 0], [((2 + 1) - 2.7 )/ 2, ((2.7 + 1) - 2 )/ 2, 0]]
        for row_pair in zip(res_two, expected):
            for pair in zip(row_pair[0].ravel(), row_pair[1]):
                self.assertAlmostEqual(pair[0], pair[1], 10)
    
    def test_jacobian_times_zvector(self):
        test_input = np.array([[5, 3, 7], [3.6, 2.7, 3.5]])
        f_eval = sparsemax(test_input)
        for i in range(test_input.shape[0]):
            f_eval_row = f_eval[i, :]
            jacobian = jacobian_sparsemax(f_eval_row)
            # eval directly
            expected = np.dot(jacobian, test_input[i, :])
            # eval indirectly
            actual = jacobian_sparsemax_times_vector(f_eval_row, test_input[i, :])
            for pair in zip(expected, actual):
                self.assertAlmostEqual(pair[0], pair[1])

    def test_loss(self):
        test_input = np.array([[5, 3, 7, 0], [3.6, 2.7, 3.5, 0]])
        actual_correct_class = [2, 2]
        expected_loss = [0, 0.3025]
        [p_array, taus] = sparsemax(test_input, return_tau=True)
        for i, pair in enumerate(zip(p_array, taus)):
            loss = loss_sparsemax(test_input[i, :], p_array[i, :], actual_correct_class[i], taus[i])
            self.assertAlmostEqual(loss, expected_loss[i])



if  __name__ == "__main__":
    t = Tests()
    t.test_evaluate_function_numpy2darrays()
    t.test_jacobian_times_zvector()
    t.test_loss()

    # make a plot to check shape
    t_seq = np.arange(-3, 3, 0.1)
    p_seq = [sparsemax(np.array([[t_seq[i], 0]]))[0, 0] for i in range(t_seq.shape[0])]
    plt.plot(t_seq, p_seq)
