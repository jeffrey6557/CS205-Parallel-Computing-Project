%%cython --annotate

from cython.parallel import prange
from cython cimport boundscheck, wraparound
import numpy as np
cimport numpy as np
import math
import scipy.io as sio
import time

cpdef addMatrix(double[:,::1] matA, double[:,::1] matB):
    cdef int i, j, row_len, col_len
    row_len = matA.shape[0]
    col_len = matA.shape[1]
    cdef double[:,::1] totalMat = np.zeros((row_len, col_len))
    for i in range(row_len):
        for j in range(col_len):
            totalMat[i,j] = matA[i,j] + matB[i,j]
    return totalMat

cpdef dotMatrix(double[:,::1] a, double[:,::1] b):
    cdef int x = a.shape[0]
    cdef int y = a.shape[1]
    cdef int z = b.shape[1]

    cdef double[:,::1] c = np.zeros((x,z))
#     cdef double val, val_a,val_b
    
    for i in range(x):
        for j in range(z): 
            for k in range(y):
                c[i,j] += a[i,k] * b[k,j]
    return c



cpdef addBias(double[:,::1] a, double[::1] b):
#     return np.insert(a,0,1,axis=1)
    cdef double[:,::1] res = np.zeros_like(a)
    cdef int c = a.shape[1]
    for i in range(c):
        res[:,i] = b
    return a + res 

cpdef linear(double[:,::1] z, double[:,::1] w, double[::1] b):
    
    return addBias(dotMatrix(z,w),b)

cpdef linearGradient(double[::1] w):
    return w
    
cpdef relu(double[:,::1] z):
    # return (z>0)*z
    
    cdef double[:,::1] ret = np.zeros_like(z)
    cdef int i, z_row = z.shape[0], z_col = z.shape[1]
    for i in range(z_row):
        for j in range(z_col):
            if z[ i,j] > 0:
                ret[i,j] = z[i,j]
    return ret

cpdef reluGradient(double[:,::1] z):
    # return (z>0)
    cdef double[:,::1] ret = np.zeros_like(z)
    cdef int i, z_row = z.shape[0], z_col = z.shape[1]
    for i in range(z_row):
        for j in range(z_col):
            if z[ i,j] > 0:
                ret[i,j] = 1
    return ret

cpdef get_loss(double[:,::1] a, double[:,::1] y):
    # return np.sum(np.square(a-y))
    cdef int i, j,a_row = a.shape[0],a_col = a.shape[1]
    cdef double loss= 0.
    for i in range(a_row):
        for j in range(a_col):
            loss += (a[i,j]-y[i,j])**2
    return loss

cpdef lossGradient(double[:,::1] a, double[:,::1] y):
    
    # return 2*a - 2*y
    cdef int i, j
    cdef double[:,::1] gradloss = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            gradloss[i,j] = 2*(a[i,j]-y[i,j])
    return gradloss # gradient of loss with respect to predicted outputs 

# cpdef diagMult(double[::1] a, double[::1] grad): 


#     # assume  a is n x 1 , grad is n x 1 = 1
#     cdef int n = grad.shape[0]
#     cdef double[:,::1] res = np.zeros((a.shape[0],1)) 

#     for j in range(a.shape[0]]):
#         for i in range(n):
#             res[j,0] += grad[i,0] * a[i,j]
        
#     return res

# cpdef compute_delta(double[:,::1] gradloss, double[:,::1] theta):
#     cdef double[:,::1] delta = np.zeros_like(gradloss)
    
    
#     return np.diag(loss_grad)


cpdef loss(double[:,::1] theta1, double[:,::1] theta2,
                     double[::1] bs,long[::1] layers,\
                  double[:,::1] inputs, double[:,::1] labels):
    
    

    # forward propagation: calculate cost
    cdef int i, inputs_nrow = inputs.shape[0]
    cdef double cost = 0.0
    
    print 'forward prop'
    # compute inputs for this layer
    cdef double[:,::1] layer_input = np.zeros_like(inputs)
#     layer_input = addBias(layer_input)
    
    layer_input = linear(layer_input,theta1,bs[:layers[0]])  # n x 3, 3 x 8 => n x 8   
    # compute outputs for this layer 
    
    cdef double[:,::1] output1 = relu(layer_input)  # n x 8 => n x 8
    print 'output1:',output1.shape[0],output1.shape[1]
    
    layer_input = output1 #addBias(output1)
    
    cdef double[:,::1] output2 = linear(layer_input,theta2,bs[layers[0]:layers[1]+layers[0]]) # n x 8, 8 x 1 => n x 1 
    print 'output2:',output2.shape[0],output2.shape[1]
    

#     layer_input = output2 #addBias(output2)
#     print 'layer 3 inputs',layer_input
#     cdef double[:,::1] output3 = linear(layer_input,theta3,bs[2]) # n x 1, 1 x 1 => n x 1
#     print 'output3:',output3.shape[1]

    cost = get_loss(output2,labels)/inputs_nrow # scalar where layer_input = predicted ouputs T x d(=1) 

    
@boundscheck(False)
@wraparound(False)
cpdef cost_function(double[:,::1] theta1, double[:,::1] theta2,
                     double[::1] bs,long[::1] layers,\
                  double[:,::1] inputs, double[:,::1] labels):
    
    

    # forward propagation: calculate cost
    cdef int i, inputs_nrow = inputs.shape[0]
    cdef double cost = 0.0
    
    print 'forward prop'
    # compute inputs for this layer
    cdef double[:,::1] layer_input = np.zeros_like(inputs)
#     layer_input = addBias(layer_input)
    
    layer_input = linear(layer_input,theta1,bs[:layers[0]])  # n x 3, 3 x 8 => n x 8   
    # compute outputs for this layer 
    
    cdef double[:,::1] output1 = relu(layer_input)  # n x 8 => n x 8
    print 'output1:',output1.shape[0],output1.shape[1]
    
    layer_input = output1 #addBias(output1)
    
    cdef double[:,::1] output2 = linear(layer_input,theta2,bs[layers[0]:layers[1]+layers[0]]) # n x 8, 8 x 1 => n x 1 
    print 'output2:',output2.shape[0],output2.shape[1]
    

#     layer_input = output2 #addBias(output2)
#     print 'layer 3 inputs',layer_input
#     cdef double[:,::1] output3 = linear(layer_input,theta3,bs[2]) # n x 1, 1 x 1 => n x 1
#     print 'output3:',output3.shape[1]

    cost = get_loss(output2,labels)/inputs_nrow # scalar where layer_input = predicted ouputs T x d(=1) 

    
    
    print 'backward prop'
    # back propagation: calculate gradiants
    cdef double[:,::1] theta1_grad = np.zeros_like(theta1)  # p x m
    cdef double[:,::1] theta2_grad = np.zeros_like(theta2)  # m x 1
    
#     print output3.shape[1],lossGradient(output2,labels).shape[1],np.diagflat(theta3)
     # 2 x 100
    cdef double[:,::1] delta2 = np.zeros((inputs_nrow,theta1.shape[1])) # n x m
    
    cdef double[:,::1] delta3 = lossGradient(output2,labels)  # n x 1
    

    # parallelize mini-batch?
    for i in range(inputs_nrow):
        
        # assume delta2[i,:] is output1[i,:] is 1 x m, theta2 is m x 1, delta3[i] is 1 x 1
        # delta2[i,:] = np.dot(np.dot(np.diagflat(output1[i,:]),theta2),delta3[i])
        #  1 x m =  1 x m , m x 1, 1 x1 
        delta2[i,:] = dotMatrix(dotMatrix(output1[i,:].reshape((1,-1))),theta2),delta3[i])


#         print 'lossGrad',lossGradient(output2,labels).shape[0],lossGradient(output2,labels).shape[1]
#         print 'diag(theta2)',np.diagflat(output1[i,:]).shape[0],np.diagflat(output1[i,:]).shape[1]
    
    # p x m     =  p x n , n x m 
    theta1_grad = np.dot(inputs.T,delta2)
    
    # m x 1    = m x n , n x 1
    theta2_grad = np.dot(output1.T,delta3)

#         print 'delta3 shape',theta3.shape[0],theta3.shape[1]
#         print 'theta2 shape',theta2.shape[0],theta2.shape[1]
        
#         print 'delta2 shape',delta2.shape[0],delta2.shape[1]
#         cdef double[:,::1] delta1 = np.dot(np.dot(theta1,delta2),np.diagflat(theta1))
#         print 'delta1',delta1.shape[1]
        
#     cdef double[:,::1] delta3 = np.multiply(lossGradient(output3,labels),theta3)
#     print 'delta3',delta3
#     cdef double[:,::1] delta2 = np.multiply(np.dot(theta3,delta3),theta2.T)
#     print 'delta2',delta2
#     cdef double[:,::1] delta1 = np.multiply(np.dot(theta2,delta2),theta1.T)
#     print 'delta1',delta1
    
#     theta3_grad = np.dot(output3, delta3)
#     theta2_grad = np.dot(output2, delta2)
#     theta1_grad = np.dot(output1, delta1)
    
    return theta2_grad,theta1_grad,delta2,delta3


    
