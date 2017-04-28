
# coding: utf-8

# In[9]:

# %matplotlib inline 
import numpy as np

from sklearn.metrics import mean_squared_error,accuracy_score,mean_absolute_error


from keras.models import Model
from keras.layers import Dense,Dropout,Input
from keras.callbacks import EarlyStopping
from keras.regularizers import l1,l2
from keras.constraints import maxnorm
from keras.optimizers import SGD,Adam,RMSprop
import pandas as pd
import time
from keras import backend as K
import os


# YOU NEED TO pip install hessianfree IF YOU DON'T HAVE THE MODULE
import hessianfree as hf
from hessianfree.loss_funcs import LossFunction
from functools import wraps



# TO TURN GPU for Keras, set devic = cuda or gpu or gpu0 like this
os.environ["THEANO_FLAGS"] = "device=cuda,openmp=1,floatX=float32" 
# TO TURN ON OPENMP
os.environ["THEANO_FLAGS"] = "device=cpu,openmp=1,floatX=float32" 


def keras_NN(n_nodes,optimizer):
    '''This function initializes and return a new neural network with regularization techniques
       
       input: 
       n_nodes: a list of units per layer like [42,24,12,1] 
       optimizer: one of the following:
        sgd = SGD
        rmsprop = RMSprop
        adagrad = Adagrad
        adadelta = Adadelta
        adam = Adam
        adamax = Adamax
        nadam = Nadam
       

       output: an object that contains these methods:
       
       model.predict(X): return predictions corresponding to X
       
       model.get_weights(): return a list of current model weights, in the order of w0,b1,w1,b1,....w4,b4
       
       model.set_weights(): takes in a list of weights in the same format as what model.get_weights() returns
       
       model.fit(X_tr,Y_tr,verbose=0,epochs=50,batch_size=1024,validation_split=0.2, callbacks=[early_stopping]): 
       
       train a model with the inputs and the specification, you can train 1 epoch;  
       and return history of loss during training (using hist.history['loss']) and validation loss if callbacks =
       [EarlyStopping(patience=5)] (using hist.history['val_loss']) 
       
    '''
    # Clear the model
    model = None
    # BUILD INPUT LAYER
    inputs = Input(shape=(n_nodes[0],))

    # CONNECT TO THE FIRST HIDDEN LAYER
    x = Dense(n_nodes[1], kernel_initializer='he_normal', 
                    kernel_regularizer=l2(0.0001),kernel_constraint = maxnorm(5), activation='relu')(inputs)
    x = Dropout(0.2)(x) # add dropout 

    # ADD SOME MORE HIDDEN LAYERS
    for i in range(2,len(n_nodes)-1):
        x = Dense(n_nodes[i],  kernel_initializer='he_normal', activation='relu',bias_initializer='he_normal',
            kernel_regularizer=l2(0.0001),kernel_constraint = maxnorm(3))(x)
        x = Dropout(0.2)(x) # add dropout 

    # OUTPUT LAYER
    predictions = Dense(1, kernel_initializer='he_normal', activation='linear')(x)

    # INITIALIZE MODEL (now you can call model.get_weights() )
    model = Model(inputs=inputs, outputs=predictions)

    # Compile model with LOSS FUNCTION and ADAM OPTIMIZER
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


###############################################################################################################

# Example OF comparing keras and Hessian Free: 

# read data and define training, validation and test set
data = np.genfromtxt('price_inputs_GS2016.csv',delimiter=',',skip_header=1)
X,ret = data[:,2:],data[:,1:2] # X means features, ret means target 
print 'shape of total X and ret:',X.shape,ret.shape

n_test = int(X.shape[0]*0.25)
N = X.shape[0] - n_test
n_val = int(N*0.2)
X_tr_temp, X_test, ret_tr_temp,ret_test = X[:-n_test],X[-n_test:],ret[:-n_test],ret[-n_test:]
X_tr,X_val,ret_tr,ret_val = X_tr_temp[:-n_val], X_tr_temp[-n_val:],ret_tr_temp[:-n_val],ret_tr_temp[-n_val:]


# define evaluation metrics
accuracy = lambda pred,truth: np.mean((pred>0)==(truth>0))
hit_ratio = lambda x,y: np.mean( ((x[1:] - x[:-1]) * (y[1:]-y[:-1]))>0 )
eval_f = [accuracy,hit_ratio,mean_squared_error,mean_absolute_error]
labels = 'accuracy,hit_ratio,mean_squared_error,mean_absolute_error'.split(',')

n_trials = 1 # run some number of trials for each model for confidence interval 

################### KERAS ONLY ######################

 
# define hyperparameters
n_nodes = [42,24,12,1] # number of units per layer
batch_size = 1024

early_stopping = EarlyStopping(patience=5)
# CHOOSE adam or adagrad 
model = keras_NN(n_nodes=n_nodes,optimizer='sgd')
model.fit(X_tr,ret_tr,verbose=0,epochs=100,batch_size=batch_size,
                 validation_data=(X_val,ret_val),callbacks=[early_stopping])
print 'After fitting on the training set for 100 epochs, keras return this weight parameter' 
print model.get_weights()


################### Hessian Free ######################



def output_loss(func):
    """Convenience decorator that takes a loss defined for the output layer
    and converts it into the more general form in terms of all layers."""

    @wraps(func)
    def wrapped_loss(self, activities, targets):
        result = [None for _ in activities[:-1]]
        result += [func(self, activities[-1], targets)]

        return result

    return wrapped_loss

class mse(LossFunction):
    
    @output_loss
    def loss(self, output, targets):
        return np.sum(np.nan_to_num(output - targets) ** 2,
                      axis=tuple(range(1, output.ndim))) / 2 /output.shape[0]

    @output_loss
    def d_loss(self, output, targets):
        return np.nan_to_num(output - targets)/output.shape[0]

    @output_loss
    def d2_loss(self, output, _):
        return np.ones_like(output)/output.shape[0]
    
def pack_weights(ff):
    '''
    input: an hessian free model
    output: a list of weight following keras' format
    ff follows this format: [(W_0,b_0),(W_1,b_1)...(W_H,b_H)]'''
    res = []
    for i in range(len(n_nodes)-1):
        weights = ff.get_weights(ff.W,(i,i+1))
        
        res.extend([np.array(weights[0]),np.array(weights[1])])
    return res

pshape = lambda a_list: [ w.shape for w in a_list]


# define hyperparameters
layers = (len(n_nodes)-1)*['ReLU'] + ['Linear'] # all relu except linear for output layer
n_nodes = [42,24,12,1] # number of units per layer
batch_size = 1024


# initialize a hessian free model with GPU use optional
ff = hf.FFNet(n_nodes,layers=layers,loss_type=mse(),
          W_init_params={ "coeff":1.0, "biases":1.0,"init_type":'gaussian'},use_GPU=0)

ff.run_epochs(X,ret,test=(X_val,ret_val),minibatch_size=1024,
                      optimizer=hf.opt.HessianFree(CG_iter=2),
                      max_epochs=50, plotting=True,print_period=None)

print 'After fitting on the training set for 100 epochs, hessian free return this weight parameter' 
print pack_weights(ff)

