import numpy as np
import pandas as pd
from mpi4py import MPI
import theano
import theano.tensor as T
from theano import function, config, shared, sandbox
import os
import particle_swarm_optimization as pso
import time

time1 = time.time()

os.environ["THEANO_FLAGS"] = "device=cpu,openmp=TRUE,floatX=float32"
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
input_col = 42
num_neutron_1 = 24
num_neutron_2 = 12
output_col=1 

DIETAG = 666
n_iteration = 100
EPSILON = 10**-5
eta = 0.1  ## learning rate
penalty_parameter=0.01

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
    f = theano.function(inputs=[x,y], outputs=[loss,L23],name='f')
    Y = np.reshape(Y,[nrow,1])
    l,pred_y = f(X,Y)
    return [np.asscalar(l), np.array([pred_y[i][0] for i in range(nrow)])]


if rank == 0:
    #data = pd.read_csv("test_data.csv",header=-1)
    data_train = np.genfromtxt('price_inputs_GS2016_train.csv',delimiter=',',skip_header=1)[:,1:]
    data_test = np.genfromtxt('price_inputs_GS2016_test.csv',delimiter=',',skip_header=1)[:,1:]
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

np.random.seed(205)
w1 = np.random.normal(loc=0,scale=1./np.sqrt(input_col),size=(input_col,num_neutron_1))
w2 = np.random.normal(loc=0,scale=1./np.sqrt(num_neutron_1),size=(num_neutron_1,num_neutron_2))
w3 = np.random.normal(loc=0,scale=1./np.sqrt(num_neutron_2),size=(num_neutron_2,output_col))
b1 = np.zeros(num_neutron_1)
b2 = np.zeros(num_neutron_2)
b3 = np.zeros(output_col)

if rank == 0:
    iteration = 0
    l_old = -2 #loss
    l_new = -1
    best_w1 = np.random.normal(loc=0,scale=1./np.sqrt(input_col),size=(input_col,num_neutron_1))
    best_w2 = np.random.normal(loc=0,scale=1./np.sqrt(num_neutron_1),size=(num_neutron_1,num_neutron_2))
    best_w3 = np.random.normal(loc=0,scale=1./np.sqrt(num_neutron_2),size=(num_neutron_2,output_col))
    best_b1 = np.zeros(num_neutron_1)
    best_b2 = np.zeros(num_neutron_2)
    best_b3 = np.zeros(output_col)
    loss_best = lossfunc(subdata[:,1:],subdata[:,0],w1,w2,w3,b1,b2,b3)
    while True:               
        status = MPI.Status()
        w1,b1,w2,b2,w3,b3 = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,status=status)
        l_new, _ = lossfunc(subdata[:,1:],subdata[:,0],w1,w2,w3,b1,b2,b3)##############
        if l_new < loss_best:#find and update best so far
            best_w1 = w1
            best_w2 = w2
            best_w3 = w3
            best_b1 = b1
            best_b2 = b2
            best_b3 = b3
            loss_best = l_new
        else:
            w1,b1,w2,b2,w3,b3 = best_w1,best_b1,best_w2,best_b2,best_w3,best_b3
        epsilon = abs(l_new - l_old)
        l_old = l_new
        if iteration == n_iteration or epsilon < EPSILON:
            break
        comm.send([w1,b1,w2,b2,w3,b3],dest=status.Get_source(),tag=0)
        print "{},{}".format(l_new,status.Get_source())
        iteration += 1

    #send message to let workers stop
    for r in range(1, size):
        comm.send([0]*6, dest=r, tag=DIETAG)
    time2 = time.time()
    loss, pred_y = lossfunc(data_test[:,1:],data_test[:,0],w1,w2,w3,b1,b2,b3)
    correct=[(pred_y[i]*data_test[i,0]>0)*1 for i in range(len(pred_y))]
    accuracy=sum(correct)/float(len(pred_y))
    print(time2-time1)
    print(accuracy)


else:
    while True:
        w1,b1,w2,b2,w3,b3 = pso.PSO(subdata[:,1:],subdata[:,0:1],[w1,b1,w2,b2,w3,b3]) ## batchsize=50, penalty parameter=1
        comm.send([w1,b1,w2,b2,w3,b3], dest=0, tag=1)
        status = MPI.Status()
        w1,b1,w2,b2,w3,b3 = comm.recv(source=0,tag=MPI.ANY_TAG,status=status)
        if status.Get_tag() == DIETAG:
            break