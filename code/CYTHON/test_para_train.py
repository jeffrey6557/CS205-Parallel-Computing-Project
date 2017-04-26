import numpy as np
from test_cost_function import para_train
def test_para_train():
    n = 100
    m = 4
    # layer_structure_cumsum = np.cumsum(np.array([m, int(m/2), 1]), dtype='int')
    # layer_structure_cumsum = np.array([m, m+int(m/2), m+int(m/2)+1], dtype='i')
    layer_structure = np.array([m+1, np.floor(m/2), 1])
    print layer_structure
    inputs_raw = np.random.randn(n, m)+2
    inputs = np.c_[np.ones(n), inputs_raw] 
    labels = (np.dot(inputs, np.array([0.1,1,2,3,4]))).reshape(-1,1)
    theta_1 = np.random.randn(m+1, int(m/2))+5
    theta_2 = np.random.randn(int(m/2), 1)+5
    weights = [theta_1, theta_2]
    grads = para_train(inputs, labels, weights, layer_structure)
    return grads
theta_grad_sum, loss=test_para_train()
print np.asarray(theta_grad_sum)
print np.asarray(loss)
