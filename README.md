
# CS 205 FINAL PROJECT REPORT 

<!-- URL: https://jeffrey6557.github.io/CS205-Parallel-Computing-Project/ -->

## Parallelizing Neural Network with Improved Performance 
Chang Liu, Greyson Liu, Kamrine Poels, Linglin Huang

## Background
Despite the availability of high-frequency stock market data, its use in forecasting stock prices is studied to a lesser extent. Similarly, despite the recent success of neural network on as a forecasting method, its power in forecasting high-frequency dynamics has been relatively overlooked. In addition, most of the studies in the literature have been focused on stock market indices instead of individual stocks. A possible explanation is the intractable computational intensity of training neural networks on the massive volume of high-frequency data of individual stocks. This motivates our study on applying parallelism to the training task and evaluate its performance to demonstrate weak and strong scaling. 


## Data
We formulate the task as a prediction problem, using lagged previous prices of individual stocks to predict future prices at the minute level. The high-frequency consolidated trade data for the US equity market comes from NYSE Trade and Quote (TAQ) database, available by the WRDS research center. 

Specifically, the inputs are price and volume information at or before minute t for all stocks except stock j. 
Technical indicators of price series includes:

 1. Exponential Moving Averages (EMA) and Moving Averages (MA)
 2. Past k-period log returns
 3. Price Trend indicators (AD, Adv, ADR)
 4. Price and returns volatility over k periods
 5. Momentum: Change in price in k periods
 6. Disparity: last available price over MA
 7. PSY: fraction of upward movement in the past k-period
 
The output is the predicted return at minute t+1 for stock j. We normalize all the input and output variables using z-score and unit norm per feature:

## Methodology and Parallelisation

### Neural Network Architecture

For the prediction method, multi-layer Artificial Neural Networks (ANN) using back-propagation algorithm has shown promising results in stock index prices compared with traditional methods [1]. Note that the traditional gradient descent algorithm of back-propagation is sequential by nature. We will therefore apply a technique that combines MPI with **three differerent parallelizable algorithms** to parallelize the training process: asynchronized multiple sub-neural networks[3] with nested parallel batch Stochastic Gradient Descent[2]. 

The initial goal of our project was to implement a two-level parallelization model by combining MPI and OpenMP. Unfortunately, developing executable code using OpenMP (via Cython) resulted in an onerous and difficult task, therefore, we opted for existing Python neural network packages that could run in parallel. Nonetheless, we describe our desired design and the design we used for our project below.

### Neural Network Architecture (hyperparameters)

We implement a **fully connected** network with:

1. L = 4 layers
2. number of neurons = 42,24,12,1; fewer neurons in deeper layers (pyramidal architecture)
3. Optimizer ADAM learning rate, other parameters such as momentum 
4. ReLu/MSE activation, linear activation for output node
5. L2 and maxnorm regularization, early stopping(patience=5), dropouts(20%)

### Parallelism Architecture

We execute data and model parallelism at two levels. Firstly, each machine (e.g. an Odyssey node) will store a Data Shard (a subset of data) and train a model replica independently and asynchronously (see Figure 1.) Each replica will fetch weights (𝑤) from the parameter server (the master node), compute ∆𝑤 with SGD, and push ∆𝑤 to the server. The parameter server updates the parameter set whenever it receives ∆𝑤 from a model replica. We implemented this level with MPI (`mpi4py` package).

# Data and Model Parallelism 

![architecture_abstract](images/architecture_abstract.png)

*Figure 1: Parallelised Neural Network Architecture [3]. Model replicas asynchronously fetch parameters 𝑤 and push ∆𝑤 to the parameter server.*

Secondly, each model replica computes ∆𝑤 by averaging the mini-batch gradients from 64 or 32 (depend on number of cores in a node) parallel threads (see Figure 2). We attempted to implement this level with OpenMP (Cython parallel module). However, we were unsuccessful with this implementation, so we used other algorithms mentioned below. 

# Parallelism in Gradient Computation
![architecture](images/architecture.png)

## Model replica algorithms

Due to the lack of success in our OpenMP algorithm, we used the algorithms listed below. 

- Stochastic Gradient Descent (SGD): stochastic approximation of the gradient descent optimization method that finds minima or maxima by iteration. 
- Adam: the learning rate is adapted for each of the parameters. Running averages of both the gradients and the second moments of the gradients are used to update parameters.
- Adaptive Gradient Algorithm (AdaGrad): modified SGD with parameter learning rate. Informally, this increases the learning rate for more sparse parameters and decreases the learning rate for less sparse ones. This strategy improves convergence performance where data is sparse. 
- Hessian-Free (Truncated Newton Method): 

SGD is implemented using the Python package [Theano](http://deeplearning.net/software/theano/), Adam and AdaGrad are implemented using [Keras](https://keras.io), and Hessian-free is applied using [hessianfree](http://pythonhosted.org/hessianfree/index.html).

# *Add figure of true architecture!!!*
<!-- ![pragmatic architecture]() -->


*Figure 2: Parallelisation in each model replica.*

## Methods and Results

### Simulations for performance analysis

We tested our two levels of parallelizations separately and then combined via simulation.

1. MPI accuracy
2. Parallelizable ANN algorithms within a model replica
    - SGD
    - Adam 
    - AdaGrad
    - Hessian-Free
3. Combined models:
    - MPI + SGD
    - MPI + Adam + AdaGrad
    - MPI + Hessian-Free

#### Performance metrics of simulations using MPI

First, we test the correctness of MPI implementation with data generated from a simple linear model. This is a reasonable *naïve* test case because ANN with zero hidden layers reduces to a linear regression if the activation function is linear.

![loss](images/simulation_MPI_loss.png)

*Figure 3: MPI simulation, loss function. The loss decreases almost exponentially as the number of epochs increases.*

![beta](images/simulation_MPI_beta.png)

*Figure 4: Convergence of parameters. All three parameters converged to their true values, respectively.*

The decrease in the loss between predicted and observed outcomes and the convergence to the true value demonstrate that our MPI algorithm operates correctly.

#### Performance Metrics of a Model Replica

##### SGD

##### Adam

![]()

##### AdaGrad

##### Hessian-Free

<!-- Secondly, we tested the performance of a single model replica using OpenMP versus CUDA implementation on predicting minute-level stock returns of Goldman Sachs in 2016. We trained a fully-connected neural network with 4 layers (# units = [42,24,12,1]) and stop training once validation is not improving for 5 epochs. For speedup experiments, epochs are set to 100. 

![loss](images/GPU_loss.png) 

*Figure 5: Convergence of loss function of different implementations (Max epochs = 100, batch size=128) *

![speedups](images/speedups.png)

*Figure 6: Speedups/thoughput (Epochs = 100) OpenMP with 32 threads and CUDA with 1 GPU machine. *

We observe that loss function converges rather quickly and has a smooth trajectory due to the relatively large size of our batches. In terms of speedups, there is a performance peak at the batch size of 128. 
-->

# Validation and Testing Methods

Because of the time series nature of the high-frequency data, we employ a walk-forward method that is suited for back-testing financial time series model. For each rolling window, we search for the best hyperparameters (#layers, nodes, etc) in the "validation timestep", and then evaluate the performance in the "testing timestep".


![backtest](images/backtest.png)
*Figure 6: Walk-forward method for time series prediction.*


The walk-forward method is implemented as follows: 

```python
# Input: define data[0 : T-1], training_size, validation_size, test_size, window_size
# Output: predicted values from t = T-training_size-validation_size : T-1 
for t in range(training_size + validation_size, T): 
    if t % test_size != 1:
        # predict on time t using a trained ensemble Neural Network;
    else:
        training_data = data[t - training_size- validation_size : t - validation_size]
        validation_data = data[t- validation_size: t]
        for i in range(N_window):
            # train the model based on a random starting point and \
            #       a bootstrapped sample of window_size from training_data;
            # cross-validate the architecture (# layer, neurons) 
            # calculate validation accuracy 
        # choose the top K models with highest accuracy to form an ensemble Neural Network; 
        # predicts on time t 
    # compute performance metrics on time t
```
Figure 8: Pseudo-code for backtesting

We search for the optimal network hyperparameters with:

1. Researcher’s guess (simple)
2. Grid Search (costly, inefficient)
3. Particle Swarm Optimization (embarassingly parallel, available in Python’s package Optunity/pyswarm)

We evaluate our model with the following metrics:

1. Effectiveness
    
    + Convergence of our model versus traditional implementation of sequential SGD 

2. Accuracies
    
    - MSE 
    - MSPE
    - Directional Accuracy (fraction of correct predictions of up and downs per model, consider thresholded on predicted values such that only large predicted values count)
    - Hit ratio

<span style="color:red"> **Add figures for exercise 2 and 4.** </span>

3. Computational cost	
    + Speedups, efficiencies, and throughputs (in Gflop/s) for different number of nodes, number of cores per core, different model size (# parameters).

<span style="color:red"> **How to get Gflop/s (time is given, cores can be deduced, how to we incorporate nodes)** </span>

## Conclusion

## References
[1] Selmi, N., Chaabene, S., & Hachicha, N. (2015). Forecasting returns on a stock market using Artificial Neural Networks and GARCH family models: Evidence of stock market S&P 500. Decision Science Letters,4(2), 203-210. doi:10.5267/j.dsl.2014.12.002

[2] Valafar, Faramarz, and Okan K. Ersoy. (1993). A Parallel Implementation of Backpropagation Neural Network on MASPAR MP-1.

[3] Dean, J., et al. (2012). Large scale distributed deep networks. Proceedings of the 25th International Conference on Neural Information Processing Systems. Lake Tahoe, Nevada, Curran Associates Inc.: 1223-1231.
