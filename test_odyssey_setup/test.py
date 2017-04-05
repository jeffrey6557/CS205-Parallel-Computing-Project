#!/usr/bin/env python
import numpy as np
import time
import omp_test
from cython import boundscheck, wraparound
from mpi4py import MPI


def test_omp(X, Y, n):
	print('Testing OMP...')
	Z_par = np.zeros((n, n))
	Z_seq = np.zeros((n, n))

	start = time.time()
	Z_par = np.array(omp_test.mmult_par(X, Y, Z_par))
	time_par = time.time() - start

	start = time.time()
	Z_seq = np.array(omp_test.mmult_seq(X, Y, Z_seq))
	time_seq = time.time() - start

	print('Correct? ', np.allclose(Z_seq, Z_par, rtol=1e-03, atol=1e-05))
	print(time_seq, time_par, 'speedup: ', time_seq/time_par)

	return(Z_seq)

def test_mpi(X, Y, n):
	TaskMaster = 0
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	sub_n = np.ceil((n+0.0)/size) #number of rows for each


	# create empty matrix to store results
	Z_sub = np.zeros((sub_n, n))

	# broad cast Y to every process
	comm.bcast(Y, root=TaskMaster)

	# scatter rows of X and Z
	sub_X = np.empty([sub_n, n], dtype='i')
	comm.scatter(X, sub_X, root=TaskMaster)

	for i in xrange(sub_n):
		for j in xrange(n):
			for k in xrange(n):
				sub_Z[i, j] += sub_X[i, k] * Y[k, j]

	if rank == TaskMaster:
		Z_mpi = np.zeros((n, n))
	comm.Gather(sub_Z, Z_mpi, root=0)
	        
	MPI.Finalize()


if __name__ == '__main__':
	n = 2**8
	X = np.random.random(size=(n, n))
	Y = np.random.random(size=(n, n))

	Z_seq = test_omp(X, Y, n)
	test_mpi(X, Y, n)
	# test_nested(X, Y, n)