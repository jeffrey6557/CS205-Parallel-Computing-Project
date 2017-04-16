import numpy as np
import pandas as pd
from mpi4py import MPI
import cost_function

# def loss(w, data):
#     X=data[:,0:3]
#     Y=data[:,3]
#     return np.dot(Y-np.dot(X,w),Y-np.dot(X,w))

# def para_train(w, subdata):
#     X=subdata[:,0:3]
#     Y=subdata[:,3]
#     return -2*np.dot(np.transpose(X),Y-np.dot(X,w))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
input_col = 2
num_neutron = 8
output_col=1 

DIETAG = 666
n_iteration = 100
EPSILON = 10**-5
eta = 0.001  ## learning rate

if rank == 0:
    data = pd.read_csv("try.csv",header=0)
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

w1 = np.ones([input_col,num_neutron])
w2 = np.ones([num_neutron,output_col])
dw1 = np.empty([input_col,num_neutron])
dw2 = np.empty([num_neutron,output_col])
bs = np.random.randn(output_col+num_neutron)
db1 = np.empty(num_neutron)
db2 = np.empty(output_col)



if rank == 0:
    iteration = 0
    l_old = -1 #loss
    while True:               
        l_new = cost_function.loss(w1,w2,bs,data[:,1:3],data[:,3:4])##############
        print "loss", l_new
        epsilon = abs(l_new - l_old)
        l_old = l_new
        if iteration > n_iteration or epsilon < EPSILON:
                break
        status = MPI.Status()
        dw1, dw2,db1,db2 = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,status=status)
        w1 = w1 - dw1*eta
        w2 = w2 - dw2*eta
        bs = bs - np.hstack((db1,db2))*eta
        comm.send([w1,w2,bs],dest=status.Get_source(),tag=0)
        print "dw1 from worker {}".format(status.Get_source())
        #print "dw", dw1
        #print 'w', w

        iteration += 1

    #send message to let workers stop
    for r in range(1, size):
        comm.send(0, dest=r, tag=DIETAG)

else:
    while True:
        dw1, dw2,db1,db2 = cost_function.cost_function(w1,w2,bs,subdata[:,1:3],subdata[:,3:4])
        comm.send([dw1,dw2,db1,db2], dest=0, tag=1)
        status = MPI.Status(),subdata[:,3]
        w1,w2,bs = comm.recv(source=0,tag=MPI.ANY_TAG,status=status)
        if status.Get_tag() == DIETAG:
            break