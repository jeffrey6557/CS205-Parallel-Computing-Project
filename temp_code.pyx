from cython.parallel import prange
from cython cimport boundscheck, wraparound
cimport numpy as np
import math
import scipy.io as sio
import time

@boundscheck(False)
@wraparound(False)
cpdef para_train(np.ndarray[np.float64_t, ndim = 2] inputs, np.ndarray[np.float64_t, ndim = 2] labels, np.ndarray[np.float64_t, ndim = 2] init_theta1, np.ndarray[np.float64_t, ndim = 2] init_theta2, double batchsize, double learningrate, int batch_iteration):
    cdef int n_cores = 64
    cdef int i
 # 	  cdef double[:,:] theta1
 #    cdef double[:,:] theta2 
 #    cdef double[:,:] theta1_tmp
 #    cdef double[:,:] theta2_tmp
    theta1 = init_theta1
    theta2 = init_theta2
    theta1_tmp = init_theta1
    theta2_tmp = init_theta2
    for i in prange(n_cores, nogil=True):
        with gil:
            for j in range(batch_iteration):
                minibatch = inputs[i * batchsize * batch_iteration + j * batchsize:i * batchsize * batch_iteration + (j + 1) * batchsize,]
                (theta1_tmp, theta2_tmp) = train(minibatch, labels, theta1_tmp, theta2_tmp, learningrate, 1)
                theta1 = theta1+theta1_tmp
                theta2 = theta2+theta2_tmp
    theta1_ret = theta1/n_cores - init_theta1
    theta2_ret = theta2/n_cores - init_theta2
    return(theta1_ret, theta2_ret)

# cpdef train(np.ndarray[np.float64_t, ndim = 2] minibatch, np.ndarray[np.float64_t, ndim = 2] labels, np.ndarray[np.float64_t, ndim = 2] theta1_tmp, np.ndarray[np.float64_t, ndim = 2] theta2_tmp, double learningrate, int batch_iteration):
#     cost, model = gradient_descent(inputs, labels, theta1, theta2, learningrate, batch_iteration)
#     return model

# cpdef gradient_descent(inputs, labels, theta1, theta2, learningrate, iteration):