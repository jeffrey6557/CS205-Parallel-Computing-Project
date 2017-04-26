
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

 1. Intra-interval proportions (IIP)
 2. Exponential Moving Averages (EMA)
 3. Price Trend indicators (AD, Adv, ADR)
 4. Others (see Appendix)

The output is the predicted price at minute t+1 for stock j.

We normalize all the input and output variables:

1. For stock price, use returns = percentage change
2. For other values, use min-max scaling: RN = (R-R_min) / (R_max – R_min) where R is the value of an input

## Methodology and Parallelisation

### Neural Network Architecture

For the prediction method, multi-layer Artificial Neural Networks (ANN) using back-propagation algorithm has shown promising results in stock index prices compared with traditional methods [1]. Note that the traditional gradient descent algorithm of back-propagation is sequential by nature. We will therefore apply a technique that combines MPI with **three differerent parallelizable algorithms** to parallelize the training process: asynchronized multiple sub-neural networks[3] with nested parallel batch Stochastic Gradient Descent[2]. 

### Neural Network Architecture (hyperparameters)

We implement a **fully connected** network with:

1. L = 1 to 10 layers
2. number of neurons = 4 to 64 per layer; fewer neurons in deeper layers (pyramidal architecture)
3. Optimizer ADAM learning rate, other parameters such as momentum 
4. ReLu/MSE activation, linear activation for output node
5. L2 regularization, early stopping, dropouts


### Parallelism Architecture

We execute data and model parallelism at two levels. Firstly, each machine (e.g. an Odyssey node) will store a Data Shard (a subset of data) and train a model replica independently and asynchronously (see Figure 1.) Each replica will fetch weights (𝑤) from the parameter server (the master node), compute ∆𝑤 with SGD, and push ∆𝑤 to the server. The parameter server updates the parameter set whenever it receives ∆𝑤 from a model replica. We implemented this level with MPI (`mpi4py` package).

# Data and Model Parallelism 

![architecture_abstract](images/architecture_abstract.png)

*Figure 1: Parallelised Neural Network Architecture [3]. Model replicas asynchronously fetch parameters 𝑤 and push ∆𝑤 to the parameter server.*

Secondly, each model replica computes ∆𝑤 by averaging the mini-batch gradients from 64 or 32 (depend on number of cores in a node) parallel threads (see Figure 2). We implemented this level with OpenMP (Cython parallel module).

# Parallelism in Gradient Computation
![architecture](images/architecture.png)
*Figure 2: Parallelisation in each model replica.*

## Experiments and Preliminary Results

### Preliminary Simulations

We tested our two levels of parallelisations separately and then combined via simulation:

##### Simulations:

1. MPI accuracy
2. Parallelizable ANN algorithms within a model replica
    i. Keras 
    ii. Hessian-Free
3. Combined models:
    i. MPI + Keras
    ii. MPI + Hessian-Free

# *Continue from here*

#### Performance metrics of simulations using MPI
First, we tested the correctness of MPI implementation with data generated from a simple linear model. We think this is a reasonable "naive" test case because an ANN without hidden layers reduces to a linear regressor when it has linear activation functions.

![loss](images/simulation_MPI_loss.png)
*Figure 3: MPI simulation, loss function*

![beta](images/simulation_MPI_betaCopy.png)
*Figure 4: MPI simulation, speed up/thoughput*

<span style="color:red"> **Can we make these images smaller to match rest of figures???** </span>

## Performance Metrics of a Model Replica
Secondly, we tested the performance of a single model replica using OpenMP versus CUDA implementation on predicting minute-level stock returns of Goldman Sachs in 2016. We trained a fully-connected neural network with 4 layers (# units = [42,24,12,1]) and stop training once validation is not improving for 5 epochs. For speedup experiments, epochs are set to 100. 

![loss](images/GPU_loss.png) 
*Convergence of loss function of different implementations (Max epochs = 100, batch size=128) *

![speedups](images/speedups.png)
*Speedups/thoughput (Epochs = 100) OpenMP with 32 threads and CUDA with 1 GPU machine. *

We observe that loss function converges rather quickly and has a smooth trajectory due to the relatively large size of our batches. In terms of speedups, there is a performance peak at the batch size of 128. 

Thirdly, we tested the combined model. <span style="color:red"> **No. Change** </span>

# Validation and Testing Methods

Because of the time series nature of the high-frequency data, we employ a walk-forward method that is suited for back-testing financial time series model. For each rolling window, we search for the best hyperparameters (#layers, nodes, etc) in the "validation timestep", and then evaluate the performance in the "testing timestep".


![backtest](images/backtest.png)
*Figure 6: Walk-forward method for time series prediction.*


The walk-foward method is implemented as follows: 

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
