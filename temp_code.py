from cython.parallel import prange
from cython cimport boundscheck, wraparound
cimport numpy as np
import math
import scipy.io as sio
import time

@boundscheck(False)
@wraparound(False)
cpdef double[:,:] para_train(double[:,:] inputs, double[:] labels, double[:,:] init_theta1, double[:,:] init_theta2, double batchsize, double learningrate, double batch_iteration):
	cdef int n_cores = 64
	cdef double[:,:] theta1 = init_theta1
    cdef double[:,:] theta2 = init_theta2
    cdef double[:] index = np.random.shuffle(len(inputs))
    cdef double[:,:] theta1_tmp
    cdef double[:,:] theta2_tmp
    for i in prange(n_cores, no_gil=True):###
        theta1_tmp = init_theta1
        theta2_tmp = init_theta2
        with gil:
            for j in range(batch_iteration):##
                minibatch = [inputs[x] for x in index[i * batchsize * iteration + j * batchsize: \
                                i * batchsize * iteration + (j + 1) * batchsize]]
                (theta1_tmp, theta2_tmp) = train(minibatch, labels, theta1_tmp, theta2_tmp, learningrate, 1)
        	theta1 += theta1_tmp
        	theta2 += theta2_tmp
    return (theta1/n_cores - init_theta1, theta2/n_cores - init_theta2)
