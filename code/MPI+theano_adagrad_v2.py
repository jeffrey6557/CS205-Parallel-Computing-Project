import numpy as np
#import pandas as pd
from mpi4py import MPI
import theano
import theano.tensor as T
from theano import function, config, shared, sandbox
import os
import keras_gradient as kg
import time

os.environ["THEANO_FLAGS"] = "device=cpu,openmp=TRUE,floatX=float32"
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
DIETAG = 666
n_iteration = 100
EPSILON = 10**-4
learning_rate = 0.1  ## learning rate
l2_rate = 0.01 # regularization l2 rate
eps = 10**-8 # error term in Adagrad

# architecture
n_nodes = [42,24,12,1]
# worker and master see param_list
param_list = []
# initialize random parameters in the order w1,w2,...b1,b2...
for i in range(len(n_nodes)-1):
    w = np.random.normal(loc = 0, scale = 1./ np.sqrt(n_nodes[i]), size = (n_nodes[i],n_nodes[i+1]) ) # 2-d array 
    param_list += [w]
for i in range(len(n_nodes)-1):   
    b = np.zeros(n_nodes[i+1])  # 1-darray 
    param_list += [b]
# master: cache has the same dimensions as parameters and gradients
cache_list = [np.zeros_like(w) for w in param_list]

def gradient(X,Y,w1,w2,w3,b1,b2,b3,batchsize):
    [nrow, ncol] = X.shape 
    index = np.random.choice(range(nrow),size=batchsize,replace=False)
    trainX = X[index,:]
    trainY = np.reshape(Y[index],[batchsize,1])
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    w01 = theano.shared(value = w1, name='w01',borrow=True)
    w12 = theano.shared(value = w2, name='w12',borrow=True)
    w23 = theano.shared(value = w3, name='w23',borrow=True)
    b01 = theano.shared(value = b1, name='b01',borrow=True)
    b12 = theano.shared(value = b2, name='b12',borrow=True)
    b23 = theano.shared(value = b3, name='b23',borrow=True)
    L01_temp = T.dot(x, w01) + b01
    L01 = L01_temp*(L01_temp>0)
    L12 = T.dot(L01, w12) + b12 
    L23 = T.dot(L12, w23) + b23
    cost = T.mean((y-L23)**2)  # the L2 penality only shows up in the master update
    # cost = loss + penalty * ((w01**2).sum() + (w12**2).sum()+(w23**2).sum())  ## L2 penalty
    dw1 = T.grad(cost=cost, wrt=w01)
    dw2 = T.grad(cost=cost, wrt=w12)
    dw3 = T.grad(cost=cost, wrt=w23)
    db1 = T.grad(cost=cost, wrt=b01)
    db2 = T.grad(cost=cost, wrt=b12)
    db3 = T.grad(cost=cost, wrt=b23)    
    train = theano.function(inputs=[x,y], outputs=[dw1,dw2,dw3,db1,db2,db3],name='train')
    gw1,gw2,gw3,gb1,gb2,gb3=train(trainX,trainY)
    return [gw1,gw2,gw3,gb1,gb2,gb3]

def lossfunc(X,Y,w1,w2,w3,b1,b2,b3):
    [nrow, ncol] = X.shape 
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    w01 = theano.shared(value = w1, name='w01',borrow=True)
    w12 = theano.shared(value = w2, name='w12',borrow=True)
    w23 = theano.shared(value = w3, name='w23',borrow=True)
    b01 = theano.shared(value = b1, name='b01',borrow=True)
    b12 = theano.shared(value = b2, name='b12',borrow=True)
    b23 = theano.shared(value = b3, name='b23',borrow=True)
    L01_temp = T.dot(x, w01) + b01
    L01 = L01_temp*(L01_temp>0)
    L12 = T.dot(L01, w12) + b12 
    L23 = T.dot(L12, w23) + b23
    loss = T.mean((y-L23)**2) 
    f = theano.function(inputs=[x,y], outputs=[loss],name='f')
    Y = np.reshape(Y,[nrow,1])
    loss = np.asscalar(f(X,Y)[0])
    return loss


if rank == 0:
    data_train = np.genfromtxt('second_level_inputs_GS2016_train.csv',delimiter=',',skip_header=1)[:,1:]
    data_test = np.genfromtxt('second_level_inputs_GS2016_test.csv',delimiter=',',skip_header=1)[:,1:]
    [nrow, ncol] = data_train.shape
    data_train=data_train.flatten()
    chunksize = int(np.ceil(0.8*nrow/(size-1)))
    len1=chunksize*ncol
    len0=nrow*ncol-len1*(size-1)
    data_train=np.hstack((data_train[(size-1)*len1:],data_train[:(size-1)*len1]))
else:
    data_train = None
    nrow = None
    ncol = None
    len0 = None
    len1 = None
    chunksize = None

nrow = comm.bcast(nrow, root=0)
ncol = comm.bcast(ncol, root=0)
len0 = comm.bcast(len0, root=0)
len1 = comm.bcast(len1, root=0)
chunksize = comm.bcast(chunksize, root=0)
comm.Barrier()

if rank == 0 :
    subdata = np.empty(len0)
else:
    subdata = np.empty(len1)

if size==3:
    comm.Scatterv([data_train,(len0,len1,len1),(0,len0,len0+len1),MPI.DOUBLE],subdata,root=0)
if size==4:
    comm.Scatterv([data_train,(len0,len1,len1,len1),(0,len0,len0+len1,len0+2*len1),MPI.DOUBLE],subdata,root=0)
if size==5:
    comm.Scatterv([data_train,(len0,len1,len1,len1,len1),(0,len0,len0+len1,len0+2*len1,len0+3*len1),MPI.DOUBLE],subdata,root=0)
if size==6:
    comm.Scatterv([data_train,(len0,len1,len1,len1,len1,len1),(0,len0,len0+len1,len0+2*len1,len0+3*len1,len0+4*len1),MPI.DOUBLE],subdata,root=0)
if size==7:
    comm.Scatterv([data_train,(len0,len1,len1,len1,len1,len1,len1),(0,len0,len0+len1,len0+2*len1,len0+3*len1,len0+4*len1,len0+5*len1),MPI.DOUBLE],subdata,root=0)    
if size==8:
    comm.Scatterv([data_train,(len0,len1,len1,len1,len1,len1,len1,len1),(0,len0,len0+len1,len0+2*len1,len0+3*len1,len0+4*len1,len0+5*len1,len0+6*len1),MPI.DOUBLE],subdata,root=0)
comm.Barrier()

if rank == 0 :
    subdata = subdata.reshape(int(len0/ncol),ncol)
else:
    subdata = subdata.reshape(int(len1/ncol),ncol)

comm.Barrier()


# master server: receive gradients, store m and v list, update and return parameters 
if rank == 0:
    iteration = 0
    l_old = -1 #loss
    mpistart_time = time.time()
    print "iteration,loss,worker"
    while True:          
        loss_time = time.time()
        l_new = lossfunc(subdata[:,1:],subdata[:,0:1],*param_list)
        epsilon = abs(l_new - l_old)
        l_old = l_new
        if iteration == n_iteration or epsilon < EPSILON:
            break
        status = MPI.Status()
        grad_list = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,status=status)
        start = time.time()
        # each grad and cache have the same dimensions as param 
        for jj, grad in enumerate(grad_list):
            cache = cache_list[jj]
            # calculate decay as a scalar 
            if jj < len(grad_list)/2 :  # when it is w 
                decay =  1 - l2_rate * learning_rate/ int(len1/ncol)
            else: # when it is b
                decay = 1.
            param_list[jj] *= decay
            # cache and grad must have the same dimensions 
            cache += grad**2 
            param_list[jj] -= learning_rate * grad / (np.sqrt(cache) + eps)
        
        comm.send(param_list,dest=status.Get_source(),tag=0)
        iteration += 1
        print "{},{},{}".format(iteration,l_new,status.Get_source())

    #send message to let workers stop
    for r in range(1, size):
        comm.send([0]*6, dest=r, tag=DIETAG)

# workers: receive parameters, return gradients
else:
    while True:
        start = time.time()
        num_batches = 20
        batchsize = 1024
        new_grad_list = [np.zeros_like(w) for w in param_list]     
        # for each batch, sum over all the gradients of batches and then 
        for ii in range(num_batches):
            grad_list = gradient(subdata[:,1:],subdata[:,0:1], *param_list, batchsize=batchsize) ## batchsize=1024
            for iii,g in enumerate(grad_list):
                new_grad_list[iii] += g
        # scale down the gradient if they are not rescaled in gradient function
        new_grad_list = [g/num_batches for g in new_grad_list]
        comm.send(new_grad_list, dest=0, tag=1)
        status = MPI.Status()
        param_list = comm.recv(source=0,tag=MPI.ANY_TAG,status=status)
        
        if status.Get_tag() == DIETAG:
            break