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

from functools import wraps
import random

import multiprocessing as mp



os.environ["THEANO_FLAGS"] = "device=cpu,openmp=1,floatX=float32"


##################################################################################################################
##################################################################################################################


def keras_NN(n_nodes,optimizer):
    '''This function initializes and return a new neural network with regularization techniques
       
       the returned object contains these methods:
       
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
        x = Dense(n_nodes[i],  kernel_initializer='he_normal', activation='relu',
            kernel_regularizer=l2(0.0001),kernel_constraint = maxnorm(3))(x)
        x = Dropout(0.2)(x) # add dropout 

    # OUTPUT LAYER
    predictions = Dense(1, kernel_initializer='he_normal', activation='linear')(x)

    # INITIALIZE MODEL (now you can call model.get_weights() )
    model = Model(inputs=inputs, outputs=predictions)

    # Compile model with LOSS FUNCTION and ADAM OPTIMIZER
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model




# demo of particle swarm optimization (PSO)

# ------------------------------------
# encode each particle as a 1 d array
def encode(weights):
    '''
    flatten a list of keras model weights (e.g.w0,b1,w1,b1,....w4,b4) into a 1-d array that represents a particl
    
    usage between encode and decode::
    position = encode(model.get_weights())
    weights = decode(position)
    '''
    position =[]
    for w in weights:
        position.extend(w.flatten().tolist())
    return np.array(position)
    
def decode(position):
    '''encode a 1-d array into a list of keras model weights (e.g.w0,b1,w1,b1,....w4,b4)
    usage between encode and decode:
    position = encode(model.get_weights())
    weights = decode(position)
    '''
    weights = []
#     print position.shape
    for i in range(len(n_nodes)-1):
        ins,outs = n_nodes[i],n_nodes[i+1]
#         print ins,outs,i*ins*(outs+1),(i+1)*ins*outs
        W = position[i*ins*outs:(i+1)*ins*outs].reshape((ins,outs))
        b = position[(i+1)*ins*outs:(i+1)*ins*outs+outs]
        weights.append(W)
        weights.append(b)
    return weights

def error(position,*args):
    weights = decode(position)
    model,X,Y = args
    model.set_weights(weights)
    return model.train_on_batch(X,Y).item()
    

# ------------------------------------


class Particle:
    def __init__(self, dim, minx, maxx, seed,model=None):
        '''if model is None, use randomized initial positions;
        otherwise, use a trained keras model weights as warm start
        '''
        self.rnd = random.Random(seed)
        
        # initialize the positions randomly; 
        self.velocity,self.position = np.zeros(dim),np.zeros(dim)
        if model is not None:
            # initialization using pretrained model weights
            early_stopping = EarlyStopping(patience=5)
            start = time.time()
            hist = model.fit(X_tr,Y_tr,verbose=0,epochs=50,batch_size=1024,validation_split=0.2,
                             callbacks=[early_stopping])
            
            self.position = encode(model.get_weights())
            
            print 'initializing a keras model %s takes'%seed,time.time()-start,'seconds'
            # personal error
            self.error = hist.history['loss'][-1] # curr error
            self.best_part_pos = self.position[:]
            self.best_part_err = [self.error] # best error
        else:
            
            # personal error
            self.error = [np.inf] # curr error
            self.best_part_pos = self.position[:]
            self.best_part_err = [self.error] # best error

        for i in range(dim):
            if model is None:
                self.position[i] = ((maxx - minx) *
                            self.rnd.random() + minx)
            self.velocity[i] = ((maxx - minx) *
                    self.rnd.random() + minx)

def Solve(max_epochs, n, dim, minx, maxx,inertia, c1,c2,warm_start = 1):
    rnd = random.Random(0)
    
    # create n random particles
    if warm_start==1:
        print 'Using warm start, fit a keras model by adagrad on the training set as the initial weight for all particles'
        swarm = [Particle(dim, minx, maxx, i,keras_NN(n_nodes,'adagrad')) for i in range(n)] 
    else:
        swarm = [Particle(dim, minx, maxx, i) for i in range(n)] 

    best_swarm_pos = np.zeros(dim) # not necess.
    best_swarm_err = [np.inf] # swarm best
    for i in range(n): # check each particle
        
        if swarm[i].error < best_swarm_err[-1]:
            best_swarm_err.append(swarm[i].error)
            best_swarm_pos = swarm[i].position[:]

    epoch = 0
    
    val_loss = []
   
    while epoch < max_epochs:
        
        # evaluate validation loss every 10 epochs
        if epoch % 10 == 0 and epoch > 1:
            print "Epoch = " + str(epoch) +\
                 " best training error = %.3f" % best_swarm_err[-1] + \
                " Validation error = %.3f" % val_loss[-1]
        
        args = (model,X_val,Y_val)
        
        val_loss.append( error(best_swarm_pos, *args))
        
        if len(val_loss)>10 and val_loss[-1] > val_loss[-10]:
            break

        for i in range(n): # process each particle

            # compute new velocity of curr particle
            for k in range(dim): 
#                 r1 = rnd.random()    # randomizations
#                 r2 = rnd.random()
                r1 = np.random.random()
                r2 = np.random.random()
                swarm[i].velocity[k] = ( (inertia * swarm[i].velocity[k]) +
                      (c1 * r1 * (swarm[i].best_part_pos[k] - 
                                  swarm[i].position[k])) +  
                      (c2 * r2 * (best_swarm_pos[k] -
                                  swarm[i].position[k])) )  
                # clip velocity as bounds
                if swarm[i].velocity[k] < minx:
                    swarm[i].velocity[k] = minx
                elif swarm[i].velocity[k] > maxx:
                    swarm[i].velocity[k] = maxx

            # compute new position using new velocity
            for k in range(dim): 
                swarm[i].position[k] += swarm[i].velocity[k]
            
            args = (model,X_tr,Y_tr)
            # compute error of new position
            swarm[i].error = error(swarm[i].position,*args)

            # is new position a new best for the particle?
            if swarm[i].error < swarm[i].best_part_err[-1]:
                swarm[i].best_part_err.append(swarm[i].error)
                swarm[i].best_part_pos = swarm[i].position[:]

            # is new position a new best overall?
            if swarm[i].error < best_swarm_err[-1]:
                best_swarm_err.append(swarm[i].error)
                best_swarm_pos = swarm[i].position[:]
        
        
        # for-each particle
        epoch += 1
    plt.plot(val_loss)
    # end while
    return best_swarm_pos,best_swarm_err
# end Solve

##########################################################################################

####################################### EXPERIMENT ###########################################



# read data 
data = np.genfromtxt('price_inputs_GS2016.csv',delimiter=',',skip_header=1)
X,Y = data[:,2:],data[:,1:2] # X means features, Y means target 


n_test = int(X.shape[0]*0.25)
N = X.shape[0] - n_test
n_val = int(N*0.2)
X_tr_temp, X_test, Y_tr_temp,Y_test = X[:-n_test],X[-n_test:],Y[:-n_test],Y[-n_test:]
X_tr,X_val,Y_tr,Y_val = X_tr_temp[:-n_val], X_tr_temp[-n_val:],Y_tr_temp[:-n_val],Y_tr_temp[-n_val:]

# define neural network
n_nodes= [X.shape[1],24,12,1]
batch_size = 1024


#########################################################################################################
# define parameters for PSO
model = keras_NN(n_nodes,'adagrad') # for warm start
dim = np.sum( n_nodes[i]*n_nodes[i+1] + n_nodes[i+1] for i in range(len(n_nodes)-1)) 


num_particles = 5
max_epochs = 100
inertia = 1.   # inertia
c1 = 1.49445 # cognitive (particle)
c2 = 3.49445 # social (swarm)

print "\nBegin particle swarm optimization using Python demo\n"
print "Goal is to solve MSE's function in " + str(dim) + " variables"
print "Setting num_particles = " + str(num_particles)
print "Setting max_epochs    = " + str(max_epochs)
print "\nStarting PSO algorithm\n"


start = time.time()
best_position,best_error = Solve(max_epochs, num_particles,
 dim, -1.0, 1.0,inertia, c1,c2,warm_start=1)
t = time.time()
print"\nPSO completed in {} seconds \n".format(t-start)
print"\nBest solution found:"
model.set_weights(decode(best_position))


print 'Evaluating on the test set:'
args = (model,X_test,Y_test)
err = error(best_position, *args)
print"Test error of best solution = %.6f" % err