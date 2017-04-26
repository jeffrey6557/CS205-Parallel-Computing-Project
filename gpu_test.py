import numpy as np
import pandas as pd
from mpi4py import MPI
input_col = 42
num_neutron_1 = 24
num_neutron_2 = 12
output_col=1 
eta=0.1

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
    train = theano.function(inputs=[x,y], outputs=[loss,dw1,dw2,dw3,db1,db2,db3],name='train',device=cuda)
    loss,gw1,gw2,gw3,gb1,gb2,gb3=train(trainX,trainY)
    return [gw1,gw2,gw3,gb1,gb2,gb3]

data = pd.read_csv("test_data.csv",header=-1)
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

cache=0
loss_vec=[]
for i in range(100):
	loss,dw1,dw2,dw3,db1,db2,db3 = gradient(data[:,0:42],data[:,43],w1,w2,w3,b1,b2,b3,50,1)
	cache += sum(dw1**2)+sum(dw2**2)+sum(dw3**2)+sum(db1**2)+sum(db2**2)+sum(db3**2)
	eta_worker=eta/np.sqrt(cache)
    w1 = w1 - dw1*eta_worker
    w2 = w2 - dw2*eta_worker
    w3 = w3 - dw3*eta_worker
    b1 = b1 - db1*eta_worker
    b2 = b2 - db2*eta_worker
    b3 = b3 - db3*eta_worker
    loss_vec.append(loss)
 print(loss_vec)