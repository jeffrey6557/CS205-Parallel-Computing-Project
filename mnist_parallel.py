import functools
import numpy as np
import math
import os
import scipy.io as sio
import time

# if os.getenv('MNISTNN_GPU') == 'yes':
#     Gpu_mode = True
# else:
#     Gpu_mode = False

# if os.getenv('MNISTNN_PARALLEL') == 'yes':
#     Distributed = True
# else:
#     Distributed = False

import theano
import theano.tensor as T

# Structure of the 3-layer neural network.
Input_layer_size = 400
Hidden_layer_size = 25
Output_layer_size = 10

# Matrix product function.  Default is to use CPU mode.
Matrix_dot = np.dot


def convert_memory_ordering_f2c(array):
    if np.isfortran(array) is True:
        return np.ascontiguousarray(array)
    else:
        return array

def load_training_data(training_file='mnistdata.mat'):
    training_data = sio.loadmat(training_file)
    inputs = training_data['X'].astype('f8')
    inputs = convert_memory_ordering_f2c(inputs)
    labels = training_data['y'].reshape(training_data['y'].shape[0])
    labels = convert_memory_ordering_f2c(labels)
    return (inputs, labels)

def load_weights(weight_file='mnistweights.mat'):
    weights = sio.loadmat(weight_file)
    theta1 = convert_memory_ordering_f2c(weights['Theta1'].astype('f8'))  # size: 25 entries, each has 401 numbers
    theta2 = convert_memory_ordering_f2c(weights['Theta2'].astype('f8'))  # size: 10 entries, each has  26 numbers
    return (theta1, theta2)

def rand_init_weights(size_in, size_out):
    epsilon_init = 0.12
    return np.random.rand(size_out, 1 + size_in) * 2 * epsilon_init - epsilon_init


def sigmoid(z):
    return 1.0 / (1 + pow(math.e, -z))


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


if Gpu_mode is True:
    def gpu_matrix_dot():
        time_start = time.time()
        x = T.matrix('x')
        y = T.matrix('y')
        z = T.dot(x, y)
        f = theano.function([x, y], z, allow_input_downcast=True)
        time_end = time.time()
        print('theano expression creation costs {} secs'.format(time_end - time_start))
        return f
else:
    def gpu_matrix_dot():
        pass

def cost_function(theta1, theta2, input_layer_size, hidden_layer_size, output_layer_size, inputs, labels, regular=0):
    input_layer = np.insert(inputs, 0, 1, axis=1)  # add bias, 5000x401
    time_start = time.time()
    hidden_layer = Matrix_dot(input_layer, theta1.T)
    hidden_layer = sigmoid(hidden_layer)
    hidden_layer = np.insert(hidden_layer, 0, 1, axis=1)  # add bias, 5000x26
    time_end = time.time()

    time_start = time.time()
    output_layer = Matrix_dot(hidden_layer, theta2.T)  # 5000x10
    output_layer = sigmoid(output_layer)
    time_end = time.time()

    # forward propagation: calculate cost
    time_start = time.time()
    cost = 0.0
    for training_index in range(len(inputs)):
        outputs = [0] * output_layer_size
        outputs[labels[training_index]-1] = 1
        for k in range(output_layer_size):
            error = -outputs[k] * math.log(output_layer[training_index][k]) - (1 - outputs[k]) * math.log(1 - output_layer[training_index][k])
            cost += error
    cost /= len(inputs)
    time_end = time.time()

    # back propagation: calculate gradiants
    time_start = time.time()
    theta1_grad = np.zeros_like(theta1)  # 25x401
    theta2_grad = np.zeros_like(theta2)  # 10x26
    for index in range(len(inputs)):
        # transform label y[i] from a number to a vector.
        outputs = np.zeros((1, output_layer_size))  # (1,10)
        outputs[0][labels[index]-1] = 1

        # calculate delta3
        delta3 = (output_layer[index] - outputs).T  # (10,1)

        # calculate delta2
        z2 = Matrix_dot(theta1, input_layer[index:index+1].T)  # (25,401) x (401,1)
        z2 = np.insert(z2, 0, 1, axis=0)  # add bias, (26,1)
        delta2 = np.multiply(
            Matrix_dot(theta2.T, delta3),  # (26,10) x (10,1)
            sigmoid_gradient(z2)  # (26,1)
        )
        delta2 = delta2[1:]  # (25,1)

        # calculate gradients of theta1 and theta2
        # (25,401) = (25,1) x (1,401)
        theta1_grad += Matrix_dot(delta2, input_layer[index:index+1])
        # (10,26) = (10,1) x (1,26)
        theta2_grad += Matrix_dot(delta3, hidden_layer[index:index+1])
    theta1_grad /= len(inputs)
    theta2_grad /= len(inputs)
    time_end = time.time()
    return cost, (theta1_grad, theta2_grad)



def gradient_descent(inputs, labels, theta1, theta2, learningrate=0.8, iteration=50):
    # if Distributed is True:
    #     if comm.rank == 0:
    #         theta1 = rand_init_weights(Input_layer_size, Hidden_layer_size)
    #         theta2 = rand_init_weights(Hidden_layer_size, Output_layer_size)
    #     else:
    #         theta1 = np.zeros((Hidden_layer_size, Input_layer_size + 1))
    #         theta2 = np.zeros((Output_layer_size, Hidden_layer_size + 1))
    #     comm.Barrier()
    #     if comm.rank == 0:
    #         time_bcast_start = time.time()
    #     comm.Bcast([theta1, MPI.DOUBLE])
    #     comm.Barrier()
    #     comm.Bcast([theta2, MPI.DOUBLE])
    #     if comm.rank == 0:
    #         time_bcast_end = time.time()
    #         print('\tBcast theta1 and theta2 uses {} secs.'.format(time_bcast_end - time_bcast_start))
    # else:
    #     theta1 = rand_init_weights(Input_layer_size, Hidden_layer_size)
    #     theta2 = rand_init_weights(Hidden_layer_size, Output_layer_size)

    cost = 0.0
    for i in range(iteration):
        time_iter_start = time.time()
        cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2, Input_layer_size, Hidden_layer_size, Output_layer_size,
        inputs, labels, regular=0)

        theta1 -= learningrate * theta1_grad
        theta2 -= learningrate * theta2_grad

        time_iter_end = time.time()
    return cost, (theta1, theta2)

def train(inputs, labels, theta1, theta2, learningrate=0.8, iteration=50):
    cost, model = gradient_descent(inputs, labels, theta1, theta2, learningrate, iteration)
    return model


def predict(model, inputs):
    theta1, theta2 = model
    a1 = np.insert(inputs, 0, 1, axis=1)  # add bias, (5000,401)
    a2 = np.dot(a1, theta1.T)  # (5000,401) x (401,25)
    a2 = sigmoid(a2)
    a2 = np.insert(a2, 0, 1, axis=1)  # add bias, (5000,26)
    a3 = np.dot(a2, theta2.T)  # (5000,26) x (26,10)
    a3 = sigmoid(a3)  # (5000,10)
    return [i.argmax()+1 for i in a3]


def para_train(inputs, labels,  init_theta1, init_theta2, batchsize=10, learningrate=0.1, batch_iteration=10):
    ################## if on the main node: n_cores = 63
    ##################  else: n_cores = 64
    n_cores = 64
    #find a subset of inputs
    theta1 = init_theta1
    theta2 = init_theta2
    index = np.random.shuffle(len(inputs))#####
    for i in prange(n_cores, no_gil=True):###
        theta1_tmp = init_theta1
        theta2_tmp = init_theta2
        with gil:
            for j in range(batch_iteration):##
                minibatch = [inputs[x] for x in index[i * batchsize * iteration + j * batchsize: \
                                i * batchsize * iteration + (j + 1) * batchsize]]
                (theta1_tmp, theta2_tmp) = train(minibatch, labels, theta1_tmp, theta2_tmp, learningrate, 1)
        theta1 += theta1_tmp
        theta2 += theta2_tmp
    return (theta1/n_cores - init_theta1, theta2/n_cores - init_theta2)





if __name__ == '__main__':
    if Gpu_mode is True:
        print('GPU mode')
        Matrix_dot = gpu_matrix_dot()
    else:
        print('CPU mode')
        Matrix_dot = np.dot

    if Distributed is True:
        print('Parallelism: yes')
    else:
        print('Parallelism: no')

    inputs, labels = load_training_data()
    model = para_train(inputs, labels, learningrate=0.1, iteration=10)
    # model = train(inputs, labels, learningrate=0.1, iteration=10)
    outputs = predict(model, inputs)
    correct_prediction = 0
    for i, predict in enumerate(outputs):
        if predict == labels[i]:
            correct_prediction += 1
    precision = float(correct_prediction) / len(labels)
    print('precision: {}'.format(precision))