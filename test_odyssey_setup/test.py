#!/usr/bin/env python
import numpy as np
import time
import omp_test
from mpi4py import MPI

def test_seq(X, Y, n):
	print('Testing seq...')
	Z_seq = np.zeros((n, n))

	Z_seq = np.array(omp_test.mmult_seq(X, Y, Z_seq))

	return(Z_seq)

def test_omp(X, Y, n):
	print('Testing OMP...')
	Z_omp = np.zeros((n, n))

	Z_omp = np.array(omp_test.mmult_par(X, Y, Z_omp))

	return(Z_omp)



def test_par(X, Y, n, comm):
	print('Testing PAR...')
	TaskMaster = 0
	rank = comm.Get_rank()
	size = comm.Get_size()
	sub_n = int(np.ceil((n+0.0)/size)) #number of rows for each


	# create empty matrix to store results
	sub_Z = np.zeros((sub_n, n))

	# if rank ==0:
	# 	print(size, n, sub_n)
	

	# broad cast Y to every process
	comm.bcast(Y, root=TaskMaster)

	# scatter rows of X and Z
	sub_X = np.empty([sub_n, n], dtype=np.float)
	comm.Scatter(X, sub_X, root=TaskMaster)

	sub_Z = np.array(omp_test.mmult_par(sub_X, Y, sub_Z))

	if rank == TaskMaster:
		Z_mpi = np.zeros((n, n))
	comm.Gather(sub_Z, Z_mpi, root=TaskMaster)
	
	# MPI.Finalize()
	# comm.Disconnect()
	return(Z_mpi)

if __name__ == '__main__':
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	if rank == 0:
		n = 2**8
		X = np.random.random(size=(n, n))
		Y = np.random.random(size=(n, n))

		start = time.time()
		Z_seq = test_seq(X, Y, n)
		time_seq = time.time() - start

		start = time.time()
		Z_omp = test_omp(X, Y, n)
		time_omp = time.time() - start
		print('OMP Correct? ', np.allclose(Z_seq, Z_omp, rtol=1e-03, atol=1e-05))
        	print(time_seq, time_omp, 'speedup: ', time_seq/time_omp)

		start = time.time()
		Z_par = test_par(X, Y, n, comm)
		time_par = time.time() - start


		print('PAR Correct? ', np.allclose(Z_seq, Z_par, rtol=1e-03, atol=1e-05))
		print(time_seq, time_par, 'speedup: ', time_seq/time_par)




