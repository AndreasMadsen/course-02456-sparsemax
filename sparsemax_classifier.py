import scipy
import numpy as np

import sparsemax
import sparsemax_loss

class SparseMaxClassifier():
	def __init__(self, X, y, learning_rate = 0.001):
		self.in_size = X.shape[1]
		self.out_size = 3 # For Iris dataset
		self.N = len(X)
		self.W = np.zeros((self.in_size, self.out_size), dtype=float)
		self.b = np.zeros(self.out_size, dtype = float)
		self.X = X
		self.y = y

		self.learning_rate = learning_rate

	def compute_gradient(self):
		z = np.dot(self.X, self.W) + self.b
		grad = np.zeros((self.N, self.out_size), dtype=float)

		for i in range(self.N):
			grad[i,:] = sparsemax_loss.sparsemax_grad(z[i,:], self.y[i])

		return (np.dot(self.X.T, grad) / self.N, np.sum(grad, axis=0) / self.N)


	def update_weights(self):
		(dW, db) = self.compute_gradient()
		self.W = self.W - self.learning_rate * dW
		self.b = self.b - self.learning_rate * db

	#def compute_loss(self):
	#	loss = sparsemax_loss.sparsemax_loss(np.dot(self.X, self.W) + self.b),


	def predict(self, x):
		return sparsemax.sparsemax(np.dot(x, self.W) + self.b)[0]