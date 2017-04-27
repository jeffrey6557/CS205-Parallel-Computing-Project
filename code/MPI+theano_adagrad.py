import numpy as np
import pandas as pd
from mpi4py import MPI
import theano
import theano.tensor as T
from theano import function, config, shared, sandbox
import os

import time

os.environ["THEANO_FLAGS"] = "device=cpu,openmp=TRUE,floatX=float32"
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
input_col = 42
num_neutron_1 = 24
num_neutron_2 = 12
output_col=1 


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




DIETAG = 666
n_iteration = 10
EPSILON = 10**-5
eta = 0.1  ## learning rate
l2_rate = 0.01 # regularization l2 rate

def gradient(trainX,trainY,w1,w2,w3,b1,b2,b3):
    [nrow, ncol] = trainX.shape 
    
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
    data = pd.read_csv("test_data.csv",header=-1)
    [nrow, ncol] = data.shape
    data=data.values.flatten()
    chunksize = int(np.ceil(nrow/size))
    len0=(nrow-(size-1)*chunksize)*ncol
    len1=chunksize*ncol

else:
    data = None
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

tuple_len=tuple([len0]+[len1]*(size-1))
tuple_loc=tuple([0] + range(len0, nrow, len1))


comm.Scatterv([data,(len0,len1,len1),(0,len0,len0+len1),MPI.DOUBLE],subdata,root=0)
#comm.Scatterv([data,(len0,len1),(0,len0),MPI.DOUBLE],subdata,root=0)
comm.Barrier()

if rank == 0 :
    data=data.reshape(nrow,ncol)
    subdata = subdata.reshape(int(len0/ncol),ncol)
else:
    subdata = subdata.reshape(int(len1/ncol),ncol)

comm.Barrier()

# w1 = np.ones([input_col,num_neutron_1])
# w2 = np.ones([num_neutron_1,num_neutron_2])
# w3 = np.ones([num_neutron_2,output_col])
# b1 = np.zeros(num_neutron_1)
# b2 = np.zeros(num_neutron_2)
# b3 = np.zeros(output_col)
# w1_temp = np.ones([input_col,num_neutron_1])
# w2_temp = np.ones([num_neutron_1,num_neutron_2])
# w3_temp = np.ones([num_neutron_2,output_col])
# b1_temp = np.zeros(num_neutron_1)
# b2_temp = np.zeros(num_neutron_2)
# b3_temp = np.zeros(output_col)
# dw1 = np.empty([input_col,num_neutron_1])
# dw2 = np.empty([num_neutron_1,num_neutron_2])
# dw3 = np.empty([num_neutron_2,output_col])
# db1 = np.empty(num_neutron_1)
# db2 = np.empty(num_neutron_2)
# db3 = np.empty(output_col)


# master server: receive gradients, store m and v list, update and return parameters 
if rank == 0:
    iteration = 0
    l_old = -1 #loss
    while True:          

        l_new = lossfunc(data[:,:42],data[:,-1],*param_list)##############
        print "Iteration {} with loss {}".format(iteration,l_new)
        epsilon = abs(l_new - l_old)
        l_old = l_new
        if iteration == n_iteration or epsilon < EPSILON:
                break
        status = MPI.Status()
        print "dw from worker {}".format(status.Get_source())

        grad_list = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,status=status)
        
        start = time.time()
        # each grad and cache have the same dimensions as param 
        for jj, grad,cache in enumerate(grad_list,cache_list):
            
            # calculate decay as a scalar 
            if jj < len(grad_list)/2 :  # when it is w 
                decay =  1 - l2_rate * learning_rate/ int(len1/ncol)
            else: # when it is b
                decay = 1.
            param_list[jj] *= decay

            # cache and grad must have the same dimensions 
            cache += grad**2 
            param_list[jj] -= learning_rate * grad / (np.sqrt(cache) + eps)
           
        print 'master ',rank,'updates parameters takes',time.time()-start
        
        comm.send(param,dest=status.Get_source(),tag=0)
        
        iteration += 1

    #send message to let workers stop
    for r in range(1, size):
        comm.send([0]*6, dest=r, tag=DIETAG)


# workers: receive parameters, return gradients
else:
    while True:
        start = time.time()
        # we can tune these such that roughly num_batches * batchsize ~= subdata.shape[0]
        
        num_batches = 1
        batchsize = subdata.shape[0] / num_batches

        new_grad_list = [np.zeros_like(w) for w in param_list]
        

        # for each batch, sum over all the gradients of batches and then 
        for ii in range(num_batches):
            batch = subdata[ ii * batchsize : min((ii + 1) * batchsize ,subdata.shape[0]) ]
            grad_list = gradient(batch[:,:42],batch[:,-1], *param_list) ## batchsize=50, penalty parameter=1
            for iii,g in enumerate(grad_list):
                new_grad_list[iii] += g

        # scale down the gradient if they are not rescaled in gradient function
        new_grad_list = [g/num_batches/batchsize for g in new_grad_list[:]]


        print 'worker ',rank,'computes gradient and takes time:',time.time()-start 
        comm.send(new_grad_list, dest=0, tag=1)
        status = MPI.Status()
        param_list = comm.recv(source=0,tag=MPI.ANY_TAG,status=status)
        
        if status.Get_tag() == DIETAG:
            break