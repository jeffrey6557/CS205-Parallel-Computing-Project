from cython.parallel import prange
from cython cimport boundscheck, wraparound
import numpy as np
cimport numpy as np
import math
import scipy.io as sio
import time
from libc.stdlib cimport malloc, free
from cpython cimport array
from cython.view cimport array as cvarray

@boundscheck(False)
@wraparound(False)
cpdef double[:,:] addMatrix(double[:,:] matA, double[:,:] matB, double[:,:] matC) nogil:
    '''matC = matA + matB'''
    cdef int nrow = matA.shape[0]
    cdef int ncol = matA.shape[1]
    for i in range(nrow):
        for j in range(ncol):
            matC[i,j] = matA[i,j] + matB[i,j]
    return matC

@boundscheck(False)
@wraparound(False)
cpdef double[:,:] dotMatrix(double[:,:] matA, double[:,:] matB, double[:,:] matC) nogil:
    '''matC = matA %*% matB, need to initialize matC as 0s'''
    cdef int x = matA.shape[0]
    cdef int y = matA.shape[1]
    cdef int z = matB.shape[1]
    for i in range(x):
        for j in range(z):
            matC[i,j] = 0
    for i in range(x):
        for j in range(y):
            for k in range(z): 
                matC[i,k] += matA[i,j] * matB[j,k]
    return matC

# @boundscheck(False)
# @wraparound(False)
# cpdef double[:] matVec(double[:,:] matA, double[:] vecB, double[:] matC) nogil:
#     '''matC = matA %*% vecB, need to initialize vecC as 0s'''
#     cdef int x = matA.shape[0]
#     cdef int y = matA.shape[1]
    
#     for i in range(x):
#             for j in range(y):
#                 matC[i] += matA[i,j] * vecB[j]
#     return matC

# @boundscheck(False)
# @wraparound(False)
# cpdef double[:,:] elemMatVec(double[:,:] matA, double[:,:] vecB, double[:,:] matC) nogil:
#     '''element-wise matrix vector multiplication: matC[i,j] = matA[i,j] * vecB[i,0]'''
#     cdef int x = matA.shape[0]
#     cdef int y = matA.shape[1]
#     cdef int z = vecB.shape[0]
#     for i in range(x):
#         for j in range(y):
#             matC[i,j] = matA[i,j] * vecB[i,0]
#     return matC

@boundscheck(False)
@wraparound(False)
cpdef double[:,:] elemMatMult(double[:,:] matA, double[:,:] matB, double[:,:] matC) nogil:
    '''element-wise matrix multiplication: matC[i,j] = matA[i,j] * matB[i,j]'''
    cdef int x = matA.shape[0]
    cdef int y = matA.shape[1]
    for i in range(x):
        for j in range(y):
            matC[i,j] = matA[i,j] * matB[i,j]
    return matC


@boundscheck(False)
@wraparound(False)
cpdef double[:,:] addBias(double[:,:] matA, double[:,:] matB) nogil:
    '''matA[i,j] = matA[i,j] + vecB[j,0], same row number'''
    cdef int nrow_a = matA.shape[0]
    cdef int ncol_a = matA.shape[1]
    cdef int nrow_b = matB.shape[0]
    if ncol_a == nrow_b:
        for i in range(nrow_a):
            for j in range(ncol_a):
                matA[i,j] = matA[i,j] + matB[j,0]
        return matA
    else:
        return None

@boundscheck(False)
@wraparound(False)
cpdef double[:,:] linear_addbias(double[:,:] z, double[:,:] w, double[:,:] b, double[:,:] l_out) nogil:
    '''l_out = dot(z, w) + b'''
    return addBias(dotMatrix(z,w,l_out),b)

@boundscheck(False)
@wraparound(False)
cpdef double[:,:] linear(double[:,:] z, double[:,:] w, double[:,:] l_out) nogil:
    '''l_out = dot(z, w)'''
    return dotMatrix(z,w,l_out)

@boundscheck(False)
@wraparound(False)
cpdef double[:,:] relu(double[:,:] z, double[:,:] r) nogil:
    '''r = max(z, 0) element-wise'''
    cdef int z_row = z.shape[0]
    cdef int z_col = z.shape[1]
    for i in range(z_row):
        for j in range(z_col):
            if z[i,j] > 0:
                r[i,j] = z[i,j]
            else:
                r[i,j] = 0
    return r

@boundscheck(False)
@wraparound(False)
cpdef double get_loss(double[:,:] a, double[:,:] y) nogil:
    '''compute SE loss: loss = sum_i((a_i-y_i)^2)'''
    # return np.sum(np.square(a-y))
    cdef int a_len = a.shape[0]
    cdef double loss= 0
    for i in range(a_len):
            loss = loss + (a[i,0]-y[i,0])**2
    return loss


@boundscheck(False)
@wraparound(False)
cpdef double[:,:] lossGradient(double[:,:] a, double[:,:] y, double[:,:] loss_grad) nogil:
    '''gradient of SE loss: loss_grad = 2*(a-y)'''
    cdef int a_len = a.shape[0]
    for i in range(a_len):
            loss_grad[i,0] = 2*(a[i,0]-y[i,0])
    return loss_grad # gradient of loss with respect to predicted outputs 



@boundscheck(False)
@wraparound(False)
cpdef double[:,:] reluGradient(double[:,:] z, double[:,:] relu_grad) nogil:
    '''gradient of relu: relu_grad = z>0'''
    cdef int i,j, z_row = z.shape[0], z_col = z.shape[1]
    for i in range(z_row):
        for j in range(z_col):
            if z[ i,j] > 0:
                relu_grad[i,j] = 1
            else:
                relu_grad[i,j] = 0
    return relu_grad



@boundscheck(False)
@wraparound(False)
cpdef double[:,:] forward_prop_naive(double[:,:] theta_1, double[:,:] theta_2, double[:] layer_structure_cumsum,
                double[:,:] inputs, double[:,:] labels, double[:,:] predictions, 
                double[:,:] layer_weighted_ave_1, double[:,:] layer_weighted_ave_2,
                double[:,:] layer_output_2
                )nogil:
    '''forward propagation: make predictions
       Parameters: theta, weights; bs, intercepts; 
                   layer_structure_cumsum, cumulative sum of number of neurons in each layer;
                   inputs, n by m matrix, where n is the number of time stamps; labels, the truth;
                   predictions, empty vector (place holder), to be updated and returned;
                   place holders for layer_weighted_ave_1,2 and layer_output_1;
       Outpus: predictions, the predicted value (output of ANN); layer_output_1
    '''
    cdef int n_inputs = inputs.shape[0]
    cdef int n_layers = int(layer_structure_cumsum.shape[0])
    cdef int n_neurons_1 = int(layer_structure_cumsum[0])#input layer
    cdef int n_neurons_2 = int(layer_structure_cumsum[1] - layer_structure_cumsum[0])#1st hidden layer
    cdef int n_neurons_3 = int(layer_structure_cumsum[2] - layer_structure_cumsum[1])#output layer
    # input layer to first hidden layer
    layer_weighted_ave_1[:,:] = linear(inputs, theta_1, layer_weighted_ave_1) # n x m, m x n_neurons_2 => n x n_neurons_2
    layer_output_2[:,:] = relu(layer_weighted_ave_1, layer_output_2) #n x n_neurons_2, outputs of layer 2
    # first hidden layer to output layer
    layer_weighted_ave_2[:,:] = linear(layer_output_2, theta_2, layer_weighted_ave_2) # n x n_neurons_2, n_neurons_2 x output_dim => n x output_dim  
    predictions[:,:] = layer_weighted_ave_2 # assuming output_dim = 1.
    return predictions

@boundscheck(False)
@wraparound(False)
cpdef double compute_loss_for_mpi(inputs, labels, weights, layer_structure):
    '''compute loss; function called by mpi; with gil (can use numpy)
       Parameters: theta, weights; bs, intercepts; 
                   layer_structure_cumsum, cumulative sum of number of neurons in each layer;
                   inputs, n by m matrix, where n is the number of time stamps; labels, the truth;
       Outpus: loss
    '''
    cdef int n_inputs = inputs.shape[0]
    theta_1 = weights[0]
    theta_2 = weights[1]
    # initialize all matrices necessary for forward_prop_naive; 
    # need to pass place holders because hard to build arrays with no gil.
    layer_structure_cumsum = np.cumsum(layer_structure)
    cdef int n_neurons_1 = int(layer_structure_cumsum[0])#input layer
    cdef int n_neurons_2 = int(layer_structure_cumsum[1] - layer_structure_cumsum[0])#1st hidden layer
    cdef int n_neurons_3 = int(layer_structure_cumsum[2] - layer_structure_cumsum[1])#output layer
    cdef double[:,:] layer_weighted_ave_1 = np.zeros((n_inputs, n_neurons_2)) #n x n_neurons_2 
    cdef double[:,:] layer_weighted_ave_2 = np.zeros((n_inputs, n_neurons_3)) #n x n_neurons_3 = n x output_dim
    cdef double[:,:] layer_output_2 = np.zeros((n_inputs, n_neurons_2)) #same as layer_weighted_ave_1
    cdef double[:,:] predictions = np.zeros((n_inputs,1))
    predictions = forward_prop_naive(theta_1, theta_2, layer_structure_cumsum, inputs, labels, predictions, 
                layer_weighted_ave_1, layer_weighted_ave_2, layer_output_2)
    cdef double loss = get_loss(predictions, labels)/n_inputs
    return loss

def test_mpi_api():
    n = 100
    m = 4
    # layer_structure_cumsum = np.cumsum(np.array([m, int(m/2), 1]), dtype='int')
    # layer_structure_cumsum = np.array([m, m+int(m/2), m+int(m/2)+1], dtype='i')
    layer_structure = np.array([m, np.floor(m/2), 1])
    print layer_structure
    cdef double[:,:] inputs_raw = np.random.randn(n, m)
    cdef double[:,:] inputs = np.c_[np.ones(n), inputs_raw] 
    cdef double[:,:] labels = (np.dot(inputs, np.array([0.1,1,2,3,4]))).reshape(-1,1)
    cdef double[:,:] theta_1 = np.random.randn(m, int(m/2))
    cdef double[:,:] theta_2 = np.random.randn(int(m/2), 1)
    weights = [theta_1, theta_2]
    loss = compute_loss_for_mpi(inputs, labels, weights, layer_structure)
    return loss








@boundscheck(False)
@wraparound(False)
cpdef double[:] train_epoch(double[:,:] theta_1, double[:,:] theta_2, double[:] layer_structure_cumsum,
                double[:,:] inputs, double[:,:] labels, double[:,:] predictions,
                double[:,:] layer_weighted_ave_1, double[:,:] layer_weighted_ave_2,
                double[:,:] layer_output_2, double[:,:] layer_output_2_grad, 
                double[:,:] theta_grad_1, double[:,:] theta_grad_2,
                double[:,:] delta_2, double[:,:] delta_3, double[:] theta_grad_all
                )nogil:
    '''forward propagation: make predictions
       Parameters: theta_1,2, weights; bs, intercepts; 
                   layer_structure_cumsum, cumulative sum of number of neurons in each layer;
                   inputs, n by m matrix, where n is the number of time stamps; labels, the truth;
                   predictions, empty vector (place holder), to be updated and returned;
                   place holders for layer_weighted_ave_1,2, layer_output_2, 
                   theta_grad_1,2(p x m, m x 1), delta_2,3(n x m, n x 1);
       Outpus: predictions, the predicted value (output of ANN)
    '''
    # forwar prop
    cdef int n_inputs = inputs.shape[0]
    cdef int m_inputs = inputs.shape[1]
    cdef int n_layers = int(layer_structure_cumsum.shape[0])
    cdef int n_neurons_1 = int(layer_structure_cumsum[0])#input layer
    cdef int n_neurons_2 = int(layer_structure_cumsum[1] - layer_structure_cumsum[0])#1st hidden layer
    cdef int n_neurons_3 = int(layer_structure_cumsum[2] - layer_structure_cumsum[1])#output layer
    # input layer to first hidden layer
    layer_weighted_ave_1[:,:] = linear(inputs, theta_1, layer_weighted_ave_1) # n x m, m x n_neurons_2 => n x n_neurons_2
#     print 'layer_weighted_ave_1',np.asarray(layer_weighted_ave_1)
    layer_output_2[:,:] = relu(layer_weighted_ave_1, layer_output_2) #n x n_neurons_2, outputs of layer 2
#     print 'layer_output_2_after',np.asarray(layer_output_2)
    # first hidden layer to output layer
#     print 'layer_weighted_ave_2',np.asarray(layer_weighted_ave_2)
    layer_weighted_ave_2[:,:] = linear(layer_output_2, theta_2, layer_weighted_ave_2) # n x n_neurons_2, n_neurons_2 x output_dim => n x output_dim  
    predictions[:,:] = layer_weighted_ave_2 # assuming output_dim = 1.
#     print 'layer_weighted_ave_2_after',np.asarray(layer_weighted_ave_2)
#     print 'theta_2',np.asarray(theta_2)
#     print 'losssssssssssss',get_loss(predictions, labels)/n_inputs
    

    # back propagation: calculate gradiants
#     print '*********************************delta_2************************************************************'
#     print 'delta3', np.array(delta_3)
    delta_3[:,:] = lossGradient(predictions, labels, delta_3)  # n x output_dim
#     print '*****predictions',np.array(predictions)
#     print '******labels',np.array(labels)
#     print 'delta3_after',np.asarray(delta_3)
    # for i in range(n_inputs):
        # assume delta2[i,:] is output1[i,:] is 1 x m, theta2 is m x 1, delta3[i] is 1 x 1
        # delta2[i,:] = np.dot(np.dot(np.diagflat(output1[i,:]),theta2),delta3[i])
        #  1 x m =  1 x m , m x 1, 1 x1 
    # delta_2[:,:] = elemMatVec(dotMatrix(layer_output_2, theta_2, delta_2), delta_3, delta_2) # n x 1
#     print '*********************************delta_2************************************************************'
#     print 'delta_2',np.asarray(delta_2)
    layer_output_2_grad[:,:] = reluGradient(layer_output_2, layer_output_2_grad)#n x n_neurons_2
    delta_2[:,:] = dotMatrix(theta_2, delta_3.T, delta_2.T).T
    delta_2[:,:] = elemMatMult(layer_output_2_grad, delta_2, delta_2) # (n_neurons_2 x 1, 1 x n).T
#     print 'delta_2_after',np.asarray(delta_2)
#     print 'layer_output_2', np.asarray(layer_output_2)
#     print 'layer_output_2_grad',np.asarray(layer_output_2_grad)
#     print 'theta_2',np.asarray(theta_2)
#     print 'delta3.T', np.asarray(delta_3.T)
    
    # input to first hidden layer: m_inputs x n_neurons_2     =  m_inputs x n_inputs , n_inputs x n_neurons_2
#     print '*********************************theta_grad_1************************************************************'
#     print 'theta_grad_1_before',np.asarray(theta_grad_1)
    theta_grad_1[:,:] = dotMatrix(inputs.T, delta_2, theta_grad_1) # m_inputs x n_neurons_2
#     print 'theta_grad_1_after',np.asarray(theta_grad_1)
#     print 'delta_2',np.asarray(delta_2)
    # first hidden layer to output: n_neurons_2 x output_dim    = n_neurons_2 x n , n x output_dim
#     print '*********************************theta_grad_2************************************************************'
#     print 'theta_grad_2',np.asarray(theta_grad_2)
    theta_grad_2[:,:] = dotMatrix(layer_output_2.T, delta_3, theta_grad_2) # n_neurons_2 x output_dim (=1)
#     print 'theta_grad_2_after',np.asarray(theta_grad_2)
#     print 'layer_output_2.T',np.asarray(layer_output_2.T)
#     print 'delta_3',np.asarray(delta_3)
#     print '*********************************grad_all************************************************************'
    theta_grad_all[:] = combine_grads(theta_grad_1, theta_grad_2, theta_grad_all)
#     print np.asarray(theta_grad_all)
    
    
    return theta_grad_all

@boundscheck(False)
@wraparound(False)
cpdef double[:] combine_grads(double[:,:] matA, double[:,:] matB, double[:] C) nogil:
    cdef int nrow_A = matA.shape[0]
    cdef int ncol_A = matA.shape[1]
    cdef int nrow_B = matB.shape[0]
    cdef int ncol_B = matB.shape[1]
    cdef int i, j
    for i in range(nrow_A):
        for j in range(ncol_A):
            C[i*ncol_A+j] = matA[i,j]
    for i in range(nrow_B):
        for j in range(ncol_B): 
            C[nrow_A*ncol_A+i*ncol_B+j] = matB[i,j]
    return C

@boundscheck(False)
@wraparound(False)
def para_train(inputs, labels, weights, layer_structure):
    N_THREADS = 2
    B_SIZE = 10
    #add intercept column
    n_inputs_all = inputs.shape[0]
    n_inputs = n_inputs_all
    cdef double[:] layer_structure_cumsum = np.cumsum(layer_structure)
    cdef int n_neurons_1 = int(layer_structure_cumsum[0])#input layer
    cdef int n_neurons_2 = int(layer_structure_cumsum[1] - layer_structure_cumsum[0])#1st hidden layer
    cdef int n_neurons_3 = int(layer_structure_cumsum[2] - layer_structure_cumsum[1])#output layer
    cdef double[:,:] theta_1 = weights[0]
    cdef double[:,:] theta_2 = weights[1]
    cdef double[:,:] predictions = np.zeros((n_inputs,1))
    cdef double[:,:] layer_weighted_ave_1 = np.zeros((n_inputs, n_neurons_2))
    cdef double[:,:] layer_weighted_ave_2 = np.zeros((n_inputs, n_neurons_3))
    cdef double[:,:] layer_output_2 = np.zeros((n_inputs, n_neurons_2))
    cdef double[:,:] layer_output_2_grad = np.zeros((n_inputs, n_neurons_2))
    cdef double[:,:] theta_grad_1 = np.zeros_like(theta_1)
    cdef double[:,:] theta_grad_2 = np.zeros_like(theta_2)
    cdef double[:,:] delta_2 = np.zeros_like(layer_weighted_ave_1)
    cdef double[:,:] delta_3 = np.zeros_like(layer_weighted_ave_2)
    #p_iters = int(n_inputs/(B_SIZE*N_THREADS))
    p_iters = 30
    
    cdef double[:] theta_grad_all = np.zeros(theta_1.shape[0]*theta_1.shape[1]+theta_2.shape[0]*theta_2.shape[1])
    cdef double[:,:] theta_grad_sum = np.zeros((p_iters,theta_grad_all.shape[0]))
    print theta_grad_all.shape[0]
    #cdef double[:,:] theta_grad_all_mat = np.zeros((5, ))
    cdef int i
    cdef int j
    cdef double[:] loss = np.zeros(p_iters)
    cdef int ip,k
    for ip in prange(1, num_threads = 1, nogil=True):
        for i in range(p_iters):
            theta_grad_all[:] = train_epoch(theta_1, theta_2, layer_structure_cumsum,
                    inputs, labels, predictions, layer_weighted_ave_1, layer_weighted_ave_2, layer_output_2, 
                    layer_output_2_grad, theta_grad_1, theta_grad_2, delta_2, delta_3, theta_grad_all)
            for j in range(theta_grad_all.shape[0]):
                theta_grad_sum[i,j] = theta_grad_all[j]
            
            for j in range(theta_1.shape[0]):
                for k in range(theta_1.shape[1]):
                    theta_1[j,k] = theta_1[j,k] - 0.000001* theta_grad_1[j,k]
            for j in range(theta_1.shape[0]):
                for k in range(theta_1.shape[1]):
                    theta_2[j,k] = theta_2[j,k] - 0.000001* theta_grad_2[j,k]
            predictions[:] = forward_prop_naive(theta_1, theta_2, layer_structure_cumsum, inputs, labels, predictions, 
                layer_weighted_ave_1, layer_weighted_ave_2, layer_output_2)
            loss[i] = get_loss(predictions, labels)
    return theta_grad_sum, loss
        


