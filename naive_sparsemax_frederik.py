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




def plot_decision_boundary(pred_func, X, y):
    #from https://github.com/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    yy = yy.astype('float32')
    xx = xx.astype('float32')
    # Predict the function value for the whole gid
    #print(pred_func(np.c_[xx.ravel(), yy.ravel()]))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])[:,0]
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)    


class onelayer_sparsemax(object):
    def __init__(self, n_features, n_classes):
        self.w = np.random.random((n_features, n_classes)) * 1  # -1 to 1

    def forward_pass(self, X):
        z = np.dot(X, self.w)
        [prob, taus] = sparsemax(z, return_tau=True)
        return prob, taus

    def update(self, X, y, prob, taus, do_update=True):
        loss_times_jacobian = []
        cost = []
        for i, row in enumerate(prob):
            loss = loss_sparsemax(X[i, :], prob[i, :], y[i], taus[i])
            jacobian = jacobian_sparsemax(prob[i, :])
            loss_times_jacobian.append(loss * jacobian.ravel())
            cost.append(loss)

        if do_update:
            self.w -= np.asarray(np.mean(loss_times_jacobian, axis=0)).reshape((2,2))

        return np.mean(cost)
    
    

if  __name__ == "__main__":
    # make a plot to check shape
    t_seq = np.arange(-3, 3, 0.1)
    p_seq = [sparsemax(np.array([[t_seq[i], 0]]))[0, 0] for i in range(t_seq.shape[0])]
    plt.plot(t_seq, p_seq)

    import sklearn.datasets
    X, y = sklearn.datasets.make_moons(300, noise=0.20)
    network = onelayer_sparsemax(len(X[0]), len(set(y)))

    X_tr = X[:100].astype('float32')
    X_val = X[100:200].astype('float32')
    X_te = X[200:].astype('float32')

    y_tr = y[:100].astype('int32')
    y_val = y[100:200].astype('int32')
    y_te = y[200:].astype('int32')

    plt.figure()
    plt.scatter(X_tr[:,0], X_tr[:,1], s=40, c=y_tr, cmap=plt.cm.BuGn)

    plot_decision_boundary(lambda x: network.forward_pass(x)[0], X_val,y_val)
    plt.title("Untrained Classifier")

    num_epochs = 1000

    train_cost, val_cost = [],[]    
    for e in range(num_epochs):
        [pred, taus] = network.forward_pass(X_tr)
        cost = network.update(X_tr, y_tr, pred, taus)

        #out = [cost, y_pred]
        train_cost += [cost]
    
        [pred, taus] = network.forward_pass(X_val)
        cost  = network.update(X_val, y_val, pred, taus, do_update=False)
        val_cost += [cost]

        if e % 100 == 0:
            print("Epoch %i, Train Cost: %0.3f\tVal Cost: %0.3f"%(e, train_cost[-1],val_cost[-1]))
     
    [pred, taus] = network.forward_pass(X_te)
    cost = network.update(X_te, y_te, pred, taus)
    print("\nTest Cost: %0.3f"%(cost))

    plot_decision_boundary(lambda x: network.forward_pass(x)[0], X_te, y_te)
    plt.title("Trained Classifier")

    epoch = np.arange(len(train_cost))
    plt.figure()
    plt.plot(epoch,train_cost,'r',epoch,val_cost,'b')
    plt.legend(['Train Loss','Val Loss'])
    plt.xlabel('Updates'), plt.ylabel('Loss')

    plt.show()


