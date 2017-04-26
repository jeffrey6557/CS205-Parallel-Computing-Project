import numpy as np
import pandas as pd
from mpi4py import MPI

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
eta = 0.01  ## learning rate

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
    L12 = T.dot(L01, w12) + b12 
    L23 = T.dot(L12, w23) + b23
    loss = T.mean((y-L23)**2)  
    cost = loss + penalty * ((w01**2).sum() + (w12**2).sum()+(w23**2).sum())  ## L2 penalty
    dw1 = T.grad(cost=cost, wrt=w01)
    dw2 = T.grad(cost=cost, wrt=w12)
    dw3 = T.grad(cost=cost, wrt=w23)
    db1 = T.grad(cost=cost, wrt=b01)
    db2 = T.grad(cost=cost, wrt=b12)
    db3 = T.grad(cost=cost, wrt=b23)    
    train = theano.function(inputs=[x,y], outputs=[dw1,dw2,dw3,db1,db2,db3],name='train',device=cuda)
    gw1,gw2,gw3,gb1,gb2,gb3=train(trainX,trainY)
    return [gw1,gw2,gw3,gb1,gb2,gb3]

def lossfunc(X,Y,w1,w2,w3,b1,b2,b3):
    Y=np.reshape(Y,[X.shape[0],1])
    L01_temp = np.dot(X, w1) + b1
    L01 = L01_temp*(L01_temp>0)
    L12 = np.dot(L01, w2) + b2 
    L23 = np.dot(L12, w3) + b3
    loss = np.mean((Y-L23)**2)  
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
comm.Barrier()

if rank == 0 :
    data=data.reshape(nrow,ncol)
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
w1_temp = np.ones([input_col,num_neutron_1])
w2_temp = np.ones([num_neutron_1,num_neutron_2])
w3_temp = np.ones([num_neutron_2,output_col])
b1_temp = np.zeros(num_neutron_1)
b2_temp = np.zeros(num_neutron_2)
b3_temp = np.zeros(output_col)
dw1 = np.empty([input_col,num_neutron_1])
dw2 = np.empty([num_neutron_1,num_neutron_2])
dw3 = np.empty([num_neutron_2,output_col])
db1 = np.empty(num_neutron_1)
db2 = np.empty(num_neutron_2)
db3 = np.empty(output_col)

if rank == 0:
    iteration = 0
    l_old = -1 #loss
    while True:               
        l_new = lossfunc(data[:,0:42],data[:,43],w1,w2,w3,b1,b2,b3)##############
        print "loss", l_new
        epsilon = abs(l_new - l_old)
        l_old = l_new
        if iteration > n_iteration or epsilon < EPSILON:
                break
        status = MPI.Status()
        dw1,dw2,dw3,db1,db2,db3 = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,status=status)
        eta_master=eta/np.sqrt(sum(dw1**2)+sum(dw2**2)+sum(dw3**2)+sum(db1**2)+sum(db2**2)+sum(db3**2))
        w1 = w1 - dw1*eta_master
        w2 = w2 - dw2*eta_master
        w3 = w3 - dw3*eta_master
        b1 = b1 - db1*eta_master
        b2 = b2 - db2*eta_master
        b3 = b3 - db3*eta_master
        comm.send([w1,w2,w3,b1,b2,b3],dest=status.Get_source(),tag=0)
        print "dw from worker {}".format(status.Get_source())
    

        iteration += 1

    #send message to let workers stop
    for r in range(1, size):
        comm.send(0, dest=r, tag=DIETAG)

else:
    while True:
        cache=0
        for j in range(100): ## 100 batch iterations in each chain
            dw1,dw2,dw3,db1,db2,db3 = gradient(subdata[:,0:42],subdata[:,43],w1_temp,w2_temp,w3_temp,b1_temp,b2_temp,b3_temp,50,1) ## batchsize=50, penalty parameter=1
            cache += sum(dw1**2)+sum(dw2**2)+sum(dw3**2)+sum(db1**2)+sum(db2**2)+sum(db3**2)
            eta_worker=eta/np.sqrt(cache)
            w1_temp = w1_temp - gw1*eta_worker
            w2_temp = w2_temp - gw2*eta_worker
            w3_temp = w3_temp - gw3*eta_worker
            b1_temp = b1_temp - gb1*eta_worker
            b2_temp = b2_temp - gb2*eta_worker
            b3_temp = b3_temp - gb3*eta_worker
        dw1 = w1_temp - w1
        dw2 = w2_temp - w2
        dw3 = w3_temp - w3
        db1 = b1_temp - b1
        db2 = b2_temp - b2
        db3 = b3_temp - b3
    comm.send([dw1,dw2,dw3,db1,db2,db3], dest=0, tag=1)
    status = MPI.Status()
    w1,w2,w3,b1,b2,b3 = comm.recv(source=0,tag=MPI.ANY_TAG,status=status)
    w1_temp = w1
    w2_temp = w2
    w3_temp = w3
    b1_temp = b1
    b2_temp = b2
    b3_temp = b3
    if status.Get_tag() == DIETAG:
        break