import numpy as np

def sparsemax(z):
	'''Implements algorithm 1
	from paper From Softmax to Sparsemax.
	Assumes a 1D numpy array as input'''
	
	K = z.size

	# Sort z and reverse
	z_sort = np.sort(z)[::-1]
	# Find k(z)
	k_z = []
	for k in range(1, K+1):
		z_k = z_sort[k-1]
		term1 = 1+k*z_k
		term2 = sum(z_sort[0:k-1])
		if term1 > term2:
			k_z.append(k)
	
	k_z_max = max(k_z) 
	# Define tau
	tau_z = (sum(z_sort[0:k_z_max])-1) / k_z_max				
	
	# Define p	
	p = np.maximum(z-tau_z, 0)

	return p, tau_z

#def sparsemaxvec(Z):
#	return np.apply_along_axis(sparsemax, 1, Z)

def sparsemax_support(z):		
	p, _ = sparsemax(z)
	s = p > 0
	s = s*1
	return s

def sparsemax_jacobian(z):
	'''Returns a n x n Jacobian matrix
	z is assumed to be N x 1'''	
	s = sparsemax_support(z)
	jacobian = np.diag(s) - np.divide(np.outer(s,s), np.sum(s))
	
	return jacobian

def sparsemax_jacobian_product(z, v):
	s = sparsemax_support(z)
	v_hat = np.divide(np.dot(s, v), np.sum(s))
	result = s * (np.subtract(v, np.dot(v_hat, np.ones(v_hat.size))))
	return result