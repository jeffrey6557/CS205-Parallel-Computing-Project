#!/usr/bin/env python
import numpy as np
import time
import omp_test
from mpi4py import MPI

nproc = MPI.COMM_WORLD.Get_size()   # Size of communicator 
iproc = MPI.COMM_WORLD.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs

if iproc == 0: print "This code is a test for mpi4py."

for i in range(0,nproc):
    MPI.COMM_WORLD.Barrier()
    if iproc == i:
        print '1. Rank %d out of %d' % (iproc,nproc)

MPI.COMM_WORLD.Barrier()

print '2. Rank %d out of %d' % (iproc,nproc)

MPI.COMM_WORLD.Barrier()
n = 2**8
X = np.random.random(size=(n, n))
Y = np.random.random(size=(n, n))
Z_seq = np.zeros((n, n))
start = time.time()
Z_seq = np.array(omp_test.mmult_seq(X, Y, Z_seq))
time_seq = time.time() - start
Z_par = np.zeros((n, n))
start = time.time()
Z_par = np.array(omp_test.mmult_par(X, Y, Z_par))
time_par = time.time() - start

print '3. Rank %d out of %d' % (iproc,nproc)
print 'PAR Correct? ', np.allclose(Z_seq, Z_par, rtol=1e-03, atol=1e-05)
print time_seq, time_par, 'speedup: ', time_seq/time_par 
        
MPI.Finalize()


