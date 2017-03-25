# CS 205 FINAL PROJECT PROPOSAL \\
Parallelizing Neural Network with Improved Performance 
 Linglin Huang, Chang Liu, Greyson Liu, Kamrine Poels

## Background
Despite the availability of high-frequency stock market data, its use in forecasting stock prices is studied to a lesser extent. Similarly, despite the recent success of neural network on as a forecasting method, its power in forecasting high-frequency dynamics has been relatively overlooked. In addition, most of the studies in the literacture have been focused on stock market indices intead of individual stocks. A possible explanation is the intractable computational intensity of training neural network on the massive volume of high-frequency data of individual stocks. This motivates our study on applying parallelism to the training task and evaluate its performance to demonstrate weak and strong scaling. 

## Methodology
We formulate the task as a prediction problem, using lagged previous prices of individual stocks to predict future prices at the minute level. The high-frequency consolidated trade data for the US equity market comes from NYSE Trade and Quote (TAQ) database, available by the WRDS research center. 
For the prediction method, multi-layer Artificial Neural Networks (ANN) using backpropagation algorithm has shown promising results in stock index prices compared with traditional methods [1]. Note that the gradient descent algorithm of backpropagation is sequential by nature. We will therefore apply two techniques in MPI and OpenMP to parallelize the training process: 1) parallelize mini-batch or Stochastic Gradient Descent method using the SIMD-BP algorithm [2] and 2) relax the optimization problem and apply alternating minimization method to solve it parallelly [3]. Then we benchmark these two approaches against the sequential version using various performance metrics on accuracy and efficiency. 

## References
[1] Selmi, N., Chaabene, S., & Hachicha, N. (2015). Forecasting returns on a stock market using Artificial Neural Networks and GARCH family models: Evidence of stock market S&P 500. Decision Science Letters,4(2), 203-210. doi:10.5267/j.dsl.2014.12.002

[2] Valafar, Faramarz, and Okan K. Ersoy. (1993). A Parallel Implementation of Backpropagation Neural Network on MASPAR MP-1.
[3] Haihao Lu & Yuanchu Dang. (2016). Alternating Minimization for ANN [PowerPoint slides]. Retrieved from http://courses.csail.mit.edu/18.337/2016/final_projects/yuanchu_dang/.


