import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import time

time1=time.time()
input_col = 42
num_neutron_1 = 24
num_neutron_2 = 12
output_col=1 
eta = 0.1
l2_rate=0.01

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
    loss = T.mean((y-L23)**2)  
    cost = loss + l2_rate * ((w01**2).sum() + (w12**2).sum()+(w23**2).sum())
    dw1 = T.grad(cost=cost, wrt=w01)
    dw2 = T.grad(cost=cost, wrt=w12)
    dw3 = T.grad(cost=cost, wrt=w23)
    db1 = T.grad(cost=cost, wrt=b01)
    db2 = T.grad(cost=cost, wrt=b12)
    db3 = T.grad(cost=cost, wrt=b23)    
    train = theano.function(inputs=[x,y], outputs=[loss,dw1,dw2,dw3,db1,db2,db3],name='train')
    l,gw1,gw2,gw3,gb1,gb2,gb3=train(trainX,trainY)
    return [l,gw1,gw2,gw3,gb1,gb2,gb3]

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


data_train = np.genfromtxt('second_level_inputs_GS2016_train.csv',delimiter=',',skip_header=1)[:,1:]
data_test = np.genfromtxt('second_level_inputs_GS2016_test.csv',delimiter=',',skip_header=1)[:,1:]
X_train=data_train[:int(np.ceil(0.8*data_train.shape[0])),1:]
Y_train=data_train[:int(np.ceil(0.8*data_train.shape[0])),0]
X_valid=data_train[int(np.ceil(0.8*data_train.shape[0])):,1:]
Y_valid=data_train[int(np.ceil(0.8*data_train.shape[0])):,0]
X_test= data_test[:,1:]
Y_test= data_test[:,0]

np.random.seed(205)
w1 = np.random.normal(loc=0,scale=1./np.sqrt(input_col),size=(input_col,num_neutron_1))
w2 = np.random.normal(loc=0,scale=1./np.sqrt(num_neutron_1),size=(num_neutron_1,num_neutron_2))
w3 = np.random.normal(loc=0,scale=1./np.sqrt(num_neutron_2),size=(num_neutron_2,output_col))
b1 = np.zeros(num_neutron_1)
b2 = np.zeros(num_neutron_2)
b3 = np.zeros(output_col)

cache_dw1 = np.zeros([input_col,num_neutron_1])
cache_dw2 = np.zeros([num_neutron_1,num_neutron_2])
cache_dw3 = np.zeros([num_neutron_2,output_col])
cache_db1 = np.zeros(num_neutron_1)
cache_db2 = np.zeros(num_neutron_2)
cache_db3 = np.zeros(output_col)
k=2000
loss_vec=[]
loss0=0
for i in range(k):
    loss1,pred_y=lossfunc(X_valid,Y_valid,w1,w2,w3,b1,b2,b3)
    loss_vec.append(loss1)
    if (abs(loss1-loss0)<10**-5):
        print(i)
        break
    loss0=loss1
    loss,dw1,dw2,dw3,db1,db2,db3 = gradient(X_train,Y_train,w1,w2,w3,b1,b2,b3,4096)
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
    b1 = b1 - np.multiply(db1, eta_b1)
    b2 = b2 - np.multiply(db2, eta_b2)
    b3 = b3 - np.multiply(db3, eta_b3)

time2=time.time()
print(loss_vec)
print(time2-time1)
loss, pred_y = lossfunc(X_test,Y_test,w1,w2,w3,b1,b2,b3)
correct=[(pred_y[i]*Y_test[i]>0)*1 for i in range(len(pred_y))]
accuracy=sum(correct)/float(len(pred_y))
print(accuracy)