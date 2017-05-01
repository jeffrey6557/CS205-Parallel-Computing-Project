import numpy as np
import pandas as pd
from mpi4py import MPI
import theano
import theano.tensor as T
from theano import function, config, shared, sandbox
import os

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
EPSILON = 10**-4
eta = 0.1  ## learning rate
penalty_parameter=0.01


def gradient(X,Y,w1,w2,w3,b1,b2,b3,batchsize,penalty):
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
    L12_temp = T.dot(L01, w12) + b12 
    L12 = L12_temp*(L12_temp>0)
    L23 = T.dot(L12, w23) + b23
    loss = T.mean((y-L23)**2)  
    cost = loss + penalty * ((w01**2).sum() + (w12**2).sum()+(w23**2).sum())  ## L2 penalty
    dw1 = T.grad(cost=cost, wrt=w01)
    dw2 = T.grad(cost=cost, wrt=w12)
    dw3 = T.grad(cost=cost, wrt=w23)
    db1 = T.grad(cost=cost, wrt=b01)
    db2 = T.grad(cost=cost, wrt=b12)
    db3 = T.grad(cost=cost, wrt=b23)    
    train = theano.function(inputs=[x,y], outputs=[loss,dw1,dw2,dw3,db1,db2,db3],name='train')
    loss,gw1,gw2,gw3,gb1,gb2,gb3=train(trainX,trainY)
    return [loss,gw1,gw2,gw3,gb1,gb2,gb3]

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
    L12_temp = T.dot(L01, w12) + b12 
    L12 = L12_temp*(L12_temp>0) 
    L23 = T.dot(L12, w23) + b23
    loss = T.mean((y-L23)**2)
    f = theano.function(inputs=[x,y], outputs=[loss],name='f')
    Y = np.reshape(Y,[nrow,1])
    loss = np.asscalar(f(X,Y)[0])
    return [loss, L23]

if rank == 0:
    #data = pd.read_csv("test_data.csv",header=-1)
    data_train = np.genfromtxt('price_inputs_GS2016_train.csv',delimiter=',',skip_header=1)[:,1:]
    data_test = np.genfromtxt('price_inputs_GS2016_test.csv',delimiter=',',skip_header=1)[:,1:]
    #X,ret = data[:,2:],data[:,1:2] # X means features, ret means target 
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

w1 = np.ones([input_col,num_neutron_1])
w2 = np.ones([num_neutron_1,num_neutron_2])
w3 = np.ones([num_neutron_2,output_col])
b1 = np.zeros(num_neutron_1)
b2 = np.zeros(num_neutron_2)
b3 = np.zeros(output_col)
dw1 = np.empty([input_col,num_neutron_1])
dw2 = np.empty([num_neutron_1,num_neutron_2])
dw3 = np.empty([num_neutron_2,output_col])
db1 = np.empty(num_neutron_1)
db2 = np.empty(num_neutron_2)
db3 = np.empty(output_col)

if rank == 0:
    fout = open("out_{}_MPI_SGD".format(size), "w")
    iteration = 0
    l_old = -1 #loss
    cache_dw1 = np.zeros([input_col,num_neutron_1])
    cache_dw2 = np.zeros([num_neutron_1,num_neutron_2])
    cache_dw3 = np.zeros([num_neutron_2,output_col])
    cache_db1 = np.zeros(num_neutron_1)
    cache_db2 = np.zeros(num_neutron_2)
    cache_db3 = np.zeros(output_col)
    while True:               
        l_new, _ = lossfunc(subdata[:,1:],subdata[:,0],w1,w2,w3,b1,b2,b3)##############
        #print "loss", l_new
        epsilon = abs(l_new - l_old)
        l_old = l_new
        if iteration == n_iteration or epsilon < EPSILON:
                break
        status = MPI.Status()
        dw1,dw2,dw3,db1,db2,db3 = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,status=status)
        cache_dw1 += dw1**2
        cache_dw2 += dw2**2
        cache_dw3 += dw3**2
        cache_db1 += db1**2
        cache_db2 += db2**2
        cache_db3 += db3**2
        eta_w1=eta/np.sqrt(cache_dw1)
        eta_w2=eta/np.sqrt(cache_dw2)
        eta_w3=eta/np.sqrt(cache_dw3)
        eta_b1=eta/np.sqrt(cache_db1)
        eta_b2=eta/np.sqrt(cache_db2)
        eta_b3=eta/np.sqrt(cache_db3)
        w1 = w1 - np.multiply(dw1, eta_w1)
        w2 = w2 - np.multiply(dw2, eta_w2)
        w3 = w3 - np.multiply(dw3, eta_w3)
        b1 = b1 - np.multiply(db1, eta_w1)
        b2 = b2 - np.multiply(db2, eta_w2)
        b3 = b3 - np.multiply(db3, eta_w3)
        comm.send([w1,w2,w3,b1,b2,b3],dest=status.Get_source(),tag=0)
        #print "dw from worker {}".format(status.Get_source())
        print "{},{}".format(l_new,status.Get_source())
        iteration += 1

    #send message to let workers stop
    for r in range(1, size):
        comm.send([0]*6, dest=r, tag=DIETAG)

    l_new, pred_y = lossfunc(data_test[:,1:],data_test[:,0],w1,w2,w3,b1,b2,b3)
    print((sum(np.dot((pred_y>0)*1,(data_test[:,0]>0)*1))+sum(np.dot((pred_y<0)*1,(data_test[:,0]<0)*1)))/data_test.shape[0])


else:
    while True:
        _,dw1,dw2,dw3,db1,db2,db3 = gradient(subdata[:,1:],subdata[:,0],w1,w2,w3,b1,b2,b3,1024,penalty_parameter) ## batchsize=50, penalty parameter=1
        comm.send([dw1,dw2,dw3,db1,db2,db3], dest=0, tag=1)
        status = MPI.Status()
        w1,w2,w3,b1,b2,b3 = comm.recv(source=0,tag=MPI.ANY_TAG,status=status)
        if status.Get_tag() == DIETAG:
            break