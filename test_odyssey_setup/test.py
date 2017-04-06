#!/usr/bin/env python
import numpy as np
import time
import omp_test
from mpi4py import MPI

TaskMaster = 0
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n = 2**8

if rank == TaskMaster:
	X = np.random.random(size=(n, n))
	Y = np.random.random(size=(n, n))
	Z_seq = np.zeros((n, n))
	Z_mpi = np.zeros((n, n))

	start = time.time()
	Z_seq = np.array(omp_test.mmult_seq(X, Y, Z_seq))
	time_seq = time.time() - start

	start = time.time()

rank = comm.Get_rank()
size = comm.Get_size()
sub_n = int(np.ceil((n+0.0)/size)) #number of rows for each process

# broad cast Y to every process
comm.bcast(Y, root=TaskMaster)

# scatter rows of X and Z
sub_X = np.empty([sub_n, n], dtype=np.float)
comm.Scatter(X, sub_X, root=TaskMaster)

# create empty matrix to store results
sub_Z = np.zeros((sub_n, n))
sub_Z = np.array(omp_test.mmult_par(sub_X, Y, sub_Z))

# gather results from all processes
comm.Gather(sub_Z, Z_mpi, root=TaskMaster)

if rank == TaskMaster:
	time_par = time.time() - start
	print('PAR Correct? ', np.allclose(Z_seq, Z_mpi, rtol=1e-03, atol=1e-05))
	print(time_seq, time_par, 'speedup: ', time_seq/time_par)




