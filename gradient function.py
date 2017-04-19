
import pandas as pd
import numpy as np
import theano
import theano.tensor as T

data = pd.read_csv("test_data.csv",header=-1)
[nrow, ncol] = data.shape
data.head(5)


data=np.array(data)
X=data[:,0:42].astype(float)
Y=data[:,43]


w1=np.random.randn(42,24)
w2=np.random.randn(24,12)
w3=np.random.randn(12,1)
b1=np.random.randn(24)
b2=np.random.randn(12)
b3=np.random.randn(1)


## regularization?
def gradient(X,Y,w1,w2,w3,b1,b2,b3,batchsize):
    '''
    1.can you return cost (a scalar) too?

    2.can you add these as arguments? 
    gradient( ..n_nodes = [24,12],mode='full',l2_rate=0.0001,max_norm=3,dropouts=0.2)
    
    
    Arguments:
    n_nodes: a list of int for hidden nodes.
    
    mode: 'full' or 'local'. Fully or locally connected architecture. 
    
    For local connection, see Figure 2 in the 205 Experimental design.docx in the Github writeup folder. 
        
        hint: is it as easy as this?
        L12 = T.dot(L01, w12) + b12 + inputs_12
        L23 = T.dot(L12, w23) + b23 + inputs_23
    
    max_norm: limit the norm(weights in each layer) < scalar 
    
    dropouts (optional): a list of floats. Probabilities of dropping out nodes in each layer, disconnected 
    in forward prop and backprop. A highly efficient regularization technique.
    
    '''
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
    # cost = loss.mean() + lambdas * ((w01**2).sum() + (w12**2).sum()+....)  ## L2 penalty
    cost = loss.mean()
    dw1 = T.grad(cost=cost, wrt=w01)
    dw2 = T.grad(cost=cost, wrt=w12)
    dw3 = T.grad(cost=cost, wrt=w23)
    db1 = T.grad(cost=cost, wrt=b01)
    db2 = T.grad(cost=cost, wrt=b12)
    db3 = T.grad(cost=cost, wrt=b23)    
    train = theano.function(inputs=[x,y], outputs=[dw1,dw2,dw3,db1,db2,db3],name='train')
    gw1,gw2,gw3,gb1,gb2,gb3=train(trainX,trainY)
    return [gw1,gw2,gw3,gb1,gb2,gb3]


gradient(X,Y,w1,w2,w3,b1,b2,b3,50)


