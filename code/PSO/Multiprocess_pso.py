#odyssey 


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


# to turn on openmp
#os.environ["THEANO_FLAGS"] = "device=cpu,openmp=1,floatX=float32"

def keras_NN(n_nodes):
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
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# demo of particle swarm optimization (PSO)

# ------------------------------------
# encode each particle as a 1 d array
def encode(weights):
    '''
    flatten a list of keras model weights (e.g.w0,b0,w1,b1,....w4,b4) into a 1-d array that represents a particl
    
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



def error(feature,target,position):
    a = feature
    weights = decode(position)
    # forward prop
#     for w in weights:
#         print w.shape,
    layers = len(weights)/2
    for l in range(layers):
#         print np.dot(X,weights[2*l])+ weights[2*l+1].reshape((1,-1))
        z = np.dot(a,weights[2*l]) + weights[2*l+1].reshape((1,-1))  # n x p * p x m + 1 x m = n x m 
        if l == layers -1:
            a = z
        else:
            a = np.maximum(0,z)
    return  np.mean( (target-a)**2 ,axis=0) 

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
            hist = model.fit(X_tr,Y_tr,verbose=0,epochs=10,batch_size=1024,validation_split=0.2,
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
        
def f_swarm(X,Y,swarm,rnd,best_swarm_pos,minx,maxx,inertia,c1,c2):
    '''a function updates a swarm 

    input: 
    update all the attributes of the swarm: position, velocity, best_part_pos, etc
    return best position and err of this swarm
    '''
#     print 'swarm %s is working' % swarm.rnd
    # compute new velocity of curr particle
    for k in range(dim): 
#         r1 = rnd.random()    # randomizations
#         r2 = rnd.random()
        r1,r2 = np.random.random(),np.random.random()
    
        swarm.velocity[k] = ( (inertia * swarm.velocity[k]) +
              (c1 * r1 * (swarm.best_part_pos[k] - 
                          swarm.position[k])) +  
              (c2 * r2 * (best_swarm_pos[k] -
                          swarm.position[k])) ) 
        
        # clip velocity as bounds
        if swarm.velocity[k] < minx:
            swarm.velocity[k] = minx
        elif swarm.velocity[k] > maxx:
            swarm.velocity[k] = maxx
            
#         # clip position as bounds
#         if swarm.position[k] < minx:
#             swarm.position[k] = minx
#         elif swarm.position[k] > maxx:
#             swarm.position[k] = maxx
    print r1,r2

    # compute new position using new velocity
#     print 'finsihed velocity'
    for k in range(dim): 
        swarm.position[k] += swarm.velocity[k]
#     print 'finsihed updating psoition'
    # compute error of new position
    swarm.error = error(X_tr,Y_tr, swarm.position)
#     print 'finished calculating error'
    # is new position a new best for the particle?
    if swarm.error < swarm.best_part_err[-1]:
        swarm.best_part_err.append(swarm.error)
        swarm.best_part_pos = swarm.position[:]
#     print 'finished upating personal best'
#     print 
    return swarm.error,swarm.position


if __name__=='__main__':

    def Solve(X_tr,Y_tr,max_epochs, n, dim, minx, maxx,inertia, c1,c2,warm_start = 1):
        rnd = random.Random(2)

        # create n random particles
        if warm_start==1:
            print 'Using warm start, fit a keras model by adam on the training set as the initial weight for all particles'
            swarms = [Particle(dim, minx, maxx, i,keras_NN(n_nodes)) for i in range(n)] 
        else:
            swarms = [Particle(dim, minx, maxx, i) for i in range(n)] 

        best_swarm_pos = [0.0 for i in range(dim)] # not necess.
        best_swarm_err = [np.inf] # swarm best

        for i in range(n): # check each particle

            if swarms[i].error < best_swarm_err[-1]:
                best_swarm_err.append(swarm[i].error)
                best_swarm_pos = swarms[i].position[:]

        epoch = 0

        
        pool = mp.Pool(processes=n)

        while epoch < max_epochs:

            if epoch % 10 == 0 and epoch > 1:
                print "Epoch = " + str(epoch) +\
                     " best error = %.3f" % best_swarm_err[-1]
                print 'Best pos',best_swarm_pos[:2]
                
    #         for i in range(n): # process each particle

            results = [pool.apply_async(f_swarm, args=(X_tr,Y_tr,swarms[i],rnd,best_swarm_pos,minx,maxx,inertia,c1,c2)) for i in range(n)]
            results = [p.get() for p in results]
#             pool.terminate()
        #     results.sort() # to sort the results by input window width
    #         print 'result of best swarm errors',results
            
            for err, pos in results:
                
            # is new position per swarm a new best overall?
            
                
                if err < best_swarm_err[-1]:
                    print 'found a better error'
                    best_swarm_err.append(err)
                    best_swarm_pos = pos[:]
            
            # for-each particle
            epoch += 1
        
        # end while
        return best_swarm_pos,best_swarm_err
    # end Solve

    ##########################################################################################

    # data = pd.read_csv("test_data.csv",header=-1).values
    # X=data[:,0:42]
    # Y=data[:,42:43]

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



    # define parameters for PSO
    model = keras_NN(n_nodes)
    dim = np.sum([np.prod(w.shape) for w in model.get_weights()])
    num_particles = 4
    max_epochs = 50
    inertia = 2.   # inertia
    c1 = 1.49445 # cognitive (particle)
    c2 = 1.49445 # social (swarm)

    print "\nBegin particle swarm optimization using Python demo\n"
    print "Goal is to solve MSE's function in " + str(dim) + " variables"
    print "Setting num_particles = " + str(num_particles)
    print "Setting max_epochs    = " + str(max_epochs)
    print "\nStarting PSO algorithm\n"

    start = time.time()
    best_position,best_error = Solve(X_tr,Y_tr,max_epochs, num_particles,
     dim, -1.0, 1.0,inertia, c1,c2,warm_start=1)
    t = time.time()-start
    print"\nTraining PSO completed in ",t, 'seconds'  
    print"\nBest solution found:"
    model.set_weights(decode(best_position))


    print 'Evaluating on the validation set:'
    
    err = error(X_val,Y_val,best_position)
    print "validation error of best solution = %.6f" % err
    print 'validation error by keras',model.evaluate(X_val,Y_val,verbose=0)
    print 'Test Error',model.evaluate(X_test,Y_test,verbose=0)
    
    print 'Other evaluation metrics'
    pred = model.predict(X_test).flatten()
    truth = Y_test.flatten()
    accuracy = lambda pred,truth: np.mean((pred>0)==(truth>0))
    hit_ratio = lambda x,y: np.mean( ((x[1:] - x[:-1]) * (y[1:]-y[:-1]))>0 )
    eval_f = [accuracy,hit_ratio,mean_squared_error,mean_absolute_error]
    labels = 'accuracy,hit_ratio,mean_squared_error,mean_absolute_error'.split(',')
    scores = [ f(pred,truth) for j,f in enumerate(eval_f) ]
    for l, v in zip(labels,scores):
        print l, v
        

