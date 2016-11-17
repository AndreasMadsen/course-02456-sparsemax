
import math
import numpy as np
import scipy

from python_reference import sparsemax
from python_reference import sparsemax_loss


class SparsemaxRegression:
    def __init__(self, input_size, output_size, observations=None,
                 regualizer=1e-1, learning_rate=1e-2,
                 random_state=None):
        self.name = 'sparsemax - numpy'
        self.random_state = random_state

        # intialize weights
        self.input_size = input_size
        self.output_size = output_size
        self.reset()

        # set hyper parameters
        self.regualizer = regualizer
        self.learning_rate = learning_rate

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def reset(self):
        self.W = scipy.stats.truncnorm.rvs(
            -2, 2, size=(self.input_size, self.output_size),
            random_state=self.random_state
        )
        self.W *= math.sqrt(2 / (self.input_size + self.output_size))
        self.b = np.zeros((1, self.output_size))

    def gradient(self, x, t):
        n = x.shape[0]
        z = np.dot(x, self.W) + self.b
        loss_grad = sparsemax_loss.grad(z, t)

        return (
            np.dot(x.T, loss_grad) / n,
            np.sum(loss_grad, axis=0) / n
        )

    def update(self, x, t, epochs=1):
        for _ in range(epochs):
            (dW, db) = self.gradient(x, t)

            self.W += - self.learning_rate * (self.regualizer * self.W + dW)
            self.b += - self.learning_rate * (self.regualizer * self.b + db)

    def loss(self, x, t):
        l2 = np.linalg.norm(self.W, 'fro')**2 + np.linalg.norm(self.b, 2)**2

        return 0.5 * self.regualizer * l2 + \
            np.mean(sparsemax_loss.forward(np.dot(x, self.W) + self.b, t))

    def predict(self, x):
        return sparsemax.forward(np.dot(x, self.W) + self.b)
