import numpy as np
import sparsemax


def sparsemax_loss(z, k):
	true_label = np.argmax(k)
	z_k = z[true_label]
	_, tau = sparsemax.sparsemax(z)
	s = sparsemax.sparsemax_support(z).astype(bool)
	return -z_k + 0.5 * np.sum(np.power(z[s],2) - np.power(tau, 2)) + 0.5


def sparsemax_grad(z,k):
	delta = np.zeros(len(z))
	delta[k] = 1
	return -delta + sparsemax.sparsemax(z)[0]




if __name__ == "__main__":
	z = np.array([2.3,2.2,2.4])
	k = 0

	print(sparsemax_grad(z,k))