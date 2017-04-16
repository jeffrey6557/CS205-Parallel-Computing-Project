
from cython.parallel import prange
from cython cimport boundscheck, wraparound
import numpy as np
cimport numpy as np
import math
import scipy.io as sio
import time

cpdef addMatrix(np.ndarray[np.float64_t, ndim = 2] matA, np.ndarray[np.float64_t, ndim = 2] matB):
    cdef int i, j, row_len, col_len
    row_len = matA.shape[0]
    col_len = matA.shape[1]
    cdef np.ndarray[np.float64_t, ndim = 2] totalMat = np.zeros((row_len, col_len))
    for i in range(row_len):
        for j in range(col_len):
            totalMat[i,j] = matA[i,j] + matB[i,j]
    return totalMat

cpdef dotMatrix(np.ndarray[np.float64_t, ndim = 2] matA, np.ndarray[np.float64_t, ndim = 2] matB):
    
    return np.dot(matA,matB)
    
#     cdef int i, j, k, a_row, a_col, b_col
#     a_row = matA.shape[0]
#     a_col = matA.shape[1]
#     b_col = matB.shape[1]
#     cdef np.ndarray[np.float64_t, ndim = 2] prodMat = addBias(matA)
#     for i in range(a_row):
#         for j in range(b_col):
#             for k in range(a_col):
#                 prodMat[i,j] = matA[i,k]*matB[k,j]
#     return prodMat

cpdef addBias(np.ndarray[np.float64_t, ndim = 2] a, np.ndarray[np.float64_t, ndim = 1] b):
#     return np.insert(a,0,1,axis=1)
    cdef np.ndarray[np.float64_t, ndim = 2] res = np.zeros_like(a)
    cdef int c = a.shape[0]
    
    for i in range(c):
        res[i,:] = b
    print res
    return a + res 

cpdef linear(np.ndarray[np.float64_t, ndim = 2] z, np.ndarray[np.float64_t, ndim = 2] w, np.ndarray[np.float64_t, ndim = 1] b):
    
    return addBias(dotMatrix(z,w),b)

# cdef linearGradient(np.ndarray[np.float64_t, ndim = 1] w):
    
cpdef relu(np.ndarray[np.float64_t, ndim = 2] z):
    return (z>0)*z
    
#     cdef np.ndarray[np.float64_t, ndim = 2] ret = np.zeros_like(z)
#     cdef int i, z_row = z.shape[0], z_col = z.shape[1]
#     for i in range(z_row):
#         for j in range(z_col):
#             if z[ i,j] > 0:
#                 ret[i,j] = z[i,j]
#     return ret

cpdef reluGradient(np.ndarray[np.float64_t, ndim = 2] z):
    return (z>0)
#     cdef np.ndarray[np.float64_t, ndim = 2] ret = np.zeros_like(z)
#     cdef int i, z_row = z.shape[0], z_col = z.shape[1]
#     for i in range(z_row):
#         for j in range(z_col):
#             if z[ i,j] > 0:
#                 ret[i,j] = 1
#     return ret

cpdef get_loss(np.ndarray[np.float64_t, ndim = 2] a, np.ndarray[np.float64_t, ndim = 2] y):
    return np.sum(np.square(a-y))
#     cdef int i, j
#     cdef double loss= 0.
#     for i in range(a.shape[0]):
#         for j in range(a.shape[1]):
#             loss += (a[i,j]-y[i,j])**2
#     return loss

cpdef lossGradient(np.ndarray[np.float64_t, ndim = 2] a, np.ndarray[np.float64_t, ndim = 2] y):
    
    return 2*a - 2*y
#     cdef int i, j
#     cdef np.ndarray[np.float64_t, ndim = 2] gradloss = np.zeros_like(a)
#     for i in range(a.shape[0]):
#         for j in range(a.shape[1]):
#             gradloss[i,j] = 2*(a[i,j]-y[i,j])
#     return gradloss # gradient of loss with respect to predicted outputs 

# cpdef diag(np.ndarray[np.float64_t, ndim = 2] grad): 
#     # grad is n x 1
#     cdef int n = grad.shape[0]
#     cdef np.ndarray[np.float64_t, ndim = 2] grad 
# #     for i in range(n):
        
#     return np.

# cpdef compute_delta(np.ndarray[np.float64_t, ndim = 2] gradloss, np.ndarray[np.float64_t, ndim = 2] theta):
#     cdef np.ndarray[np.float64_t, ndim = 2] delta = np.zeros_like(gradloss)
    
    
#     return np.diag(loss_grad)
    
cpdef loss(np.ndarray[np.float64_t, ndim = 2] theta1, np.ndarray[np.float64_t, ndim = 2] theta2,\
                 np.ndarray[np.float64_t, ndim = 2] bs,
                  np.ndarray[np.float64_t, ndim = 2] inputs, np.ndarray[np.float64_t, ndim = 2] labels):
    
    

    # forward propagation: calculate cost
    cdef int i, inputs_nrow = inputs.shape[0]
    cdef double cost = 0.0
    
    print 'forward prop'
    # compute inputs for this layer
    cdef np.ndarray[np.float64_t, ndim = 2] layer_input = np.zeros_like(inputs)
#     layer_input = addBias(layer_input)
    
    layer_input = linear(layer_input,theta1,bs[0])  # n x 3, 3 x 8 => n x 8   
    # compute outputs for this layer 
    
    cdef np.ndarray[np.float64_t, ndim = 2] output1 = relu(layer_input)  # n x 8 => n x 8
    print 'output1:',output1.shape[0],output1.shape[1]
    
    layer_input = output1 #addBias(output1)
    
    cdef np.ndarray[np.float64_t, ndim = 2] output2 = linear(layer_input,theta2,bs[1]) # n x 8, 8 x 1 => n x 1 
    print 'output2:',output2.shape[0],output2.shape[1]
    

#     layer_input = output2 #addBias(output2)
#     print 'layer 3 inputs',layer_input
#     cdef np.ndarray[np.float64_t, ndim = 2] output3 = linear(layer_input,theta3,bs[2]) # n x 1, 1 x 1 => n x 1
#     print 'output3:',output3.shape[1]

    cost = get_loss(output2,labels)/inputs_nrow
    return cost

@boundscheck(False)
@wraparound(False)
cpdef cost_function(np.ndarray[np.float64_t, ndim = 2] theta1, np.ndarray[np.float64_t, ndim = 2] theta2,
                 np.ndarray[np.float64_t, ndim = 1] bs, np.ndarray[np.int_t, ndim = 1] layers,
                  np.ndarray[np.float64_t, ndim = 2] inputs, np.ndarray[np.float64_t, ndim = 2] labels):
    

    # forward propagation: calculate cost
    cdef int i, inputs_nrow = inputs.shape[0]
    cdef double cost = 0.0
#     cdef np.ndarray[np.float64_t, ndim = 2] bs = bs1.repeat(inputs_nrow).reshape((-1,inputs_nrow))
    
    print 'forward prop'
    # compute inputs for this layer
    cdef np.ndarray[np.float64_t, ndim = 2] layer_input = np.zeros_like(inputs)
#     layer_input = addBias(layer_input)
    
    layer_input = linear(layer_input,theta1,bs[:layers[0]])  # n x 3, 3 x 8 => n x 8   
    # compute outputs for this layer 
    print 'bs[0]',bs[layers[0]:]
    
    cdef np.ndarray[np.float64_t, ndim = 2] output1 = relu(layer_input)  # n x 8 => n x 8
    print 'output1:',output1.shape[0],output1.shape[1]
    
    layer_input = output1 #addBias(output1)
    
    cdef np.ndarray[np.float64_t, ndim = 2] output2 = linear(layer_input,theta2,bs[layers[0]:layers[1]+layers[0]]) # n x 8, 8 x 1 => n x 1 
    print 'output2:',output2.shape[0],output2.shape[1]
    print 'bs[1]',bs[layers[0]:layers[1]+layers[0]]

#     layer_input = output2 #addBias(output2)
#     print 'layer 3 inputs',layer_input
#     cdef np.ndarray[np.float64_t, ndim = 2] output3 = linear(layer_input,theta3,bs[2]) # n x 1, 1 x 1 => n x 1
#     print 'output3:',output3.shape[1]

    cost = get_loss(output2,labels)/inputs_nrow
    # scalar where layer_input = predicted ouputs T x d(=1) 

    
    
    print 'backward prop'
    # back propagation: calculate gradiants
    cdef np.ndarray[np.float64_t, ndim = 2] theta1_grad = np.zeros_like(theta1)  # 25x401
    cdef np.ndarray[np.float64_t, ndim = 2] theta2_grad = np.zeros_like(theta2)  # 10x26
    
#     print output3.shape[1],lossGradient(output2,labels).shape[1],np.diagflat(theta3)
     # 2 x 100
    cdef np.ndarray[np.float64_t, ndim = 2] delta2 = np.zeros((inputs_nrow,theta1.shape[1]))
    
    cdef np.ndarray[np.float64_t, ndim = 2] delta3 = lossGradient(output2,labels)  # n x 1
    
    for i in range(inputs_nrow):
        
        delta2[i,:] = np.dot(np.dot(np.diagflat(output1[i,:]),theta2),delta3[i])
        
#         print 'lossGrad',lossGradient(output2,labels).shape[0],lossGradient(output2,labels).shape[1]
#         print 'diag(theta2)',np.diagflat(output1[i,:]).shape[0],np.diagflat(output1[i,:]).shape[1]
    
    theta1_grad = np.dot(inputs.T,delta2)
    
    
    theta2_grad = np.dot(output1.T,delta3)

#         print 'delta3 shape',theta3.shape[0],theta3.shape[1]
#         print 'theta2 shape',theta2.shape[0],theta2.shape[1]
        
#         print 'delta2 shape',delta2.shape[0],delta2.shape[1]
#         cdef np.ndarray[np.float64_t, ndim = 2] delta1 = np.dot(np.dot(theta1,delta2),np.diagflat(theta1))
#         print 'delta1',delta1.shape[1]
        
#     cdef np.ndarray[np.float64_t, ndim = 2] delta3 = np.multiply(lossGradient(output3,labels),theta3)
#     print 'delta3',delta3
#     cdef np.ndarray[np.float64_t, ndim = 2] delta2 = np.multiply(np.dot(theta3,delta3),theta2.T)
#     print 'delta2',delta2
#     cdef np.ndarray[np.float64_t, ndim = 2] delta1 = np.multiply(np.dot(theta2,delta2),theta1.T)
#     print 'delta1',delta1
    
#     theta3_grad = np.dot(output3, delta3)
#     theta2_grad = np.dot(output2, delta2)
#     theta1_grad = np.dot(output1, delta1)
    
    return theta1_grad,theta2_grad,delta2,delta3