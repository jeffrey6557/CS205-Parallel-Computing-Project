from cython.parallel import prange
cimport cython
cimport openmp

@cython.boundscheck(False)
@cython.wraparound(False)
def mmult_par(double[:, :] X, 
	double[:, :] Y, double[:, :] Z):
	
	cdef int I = X.shape[0], J = Y.shape[0], K = X.shape[1]
	cdef int i, j, k

	for i in prange(I, nogil=True):
		with gil:
			for j in range(J):
				for k in range(K):
					Z[i,j] += X[i,k] * Y[k,j]

	# for i in range(N):
	# 		for j in range(N):
	# 			for k in prange(N, nogil=True):
	# 				Z[i,j] += X[i,k] * Y[k,j]
	return(Z)

@cython.boundscheck(False)
@cython.wraparound(False)
def mmult_seq(double[:, :] X, 
	double[:, :] Y, double[:, :] Z):
	
	cdef int I = X.shape[0], J = Y.shape[0], K = X.shape[1]
	cdef int i, j, k

	for i in range(I):
			for j in range(J):
				for k in range(K):
					Z[i,j] += X[i,k] * Y[k,j]
	return(Z)
