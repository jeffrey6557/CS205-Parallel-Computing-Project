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

def get_keras_gradients(X,Y, weights,model):
    '''
    input : X, Y, weights (a list in the order of w1,b1...wn,bn )
    output: a list of gradients with the same dimension as weights
    
    '''
    
    model.set_weights(weights)
    
    input_tensors = [model.inputs[0], # input data
                      model.sample_weights[0],
                     model.targets[0], # labels
                     K.learning_phase() # train or test mode
    ]

    gradients = model.optimizer.get_gradients(model.total_loss,model.trainable_weights)
    get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    
    inputs = [X, # X
              np.ones(X.shape[0]), # sample weights
              Y, # y
              1 # learning phase in TRAINING mode 
        ]

    return get_gradients(inputs)




# read data and define training, validation and test set
data = np.genfromtxt('price_inputs_GS2016.csv',delimiter=',',skip_header=1)
X,ret = data[:,2:],data[:,1:2] # X means features, ret means target 
print 'shape of total X and ret:',X.shape,ret.shape

n_test = int(X.shape[0]*0.25)
N = X.shape[0] - n_test
n_val = int(N*0.2)
X_tr_temp, X_test, ret_tr_temp,ret_test = X[:-n_test],X[-n_test:],ret[:-n_test],ret[-n_test:]
X_tr,X_val,ret_tr,ret_val = X_tr_temp[:-n_val], X_tr_temp[-n_val:],ret_tr_temp[:-n_val],ret_tr_temp[-n_val:]


n_nodes = [42,24,12,1] # number of units per layer


model = keras_NN(n_nodes=n_nodes,optimizer='adagrad')

# change this to what master gives
current_weight = model.get_weights() 

get_keras_gradients(X_tr,ret_tr, current_weight, model)
