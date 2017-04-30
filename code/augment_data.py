import numpy as np

def augment_data(data,n_boot = 500):
    
    res = data.copy()
    for i in range(n_boot):
        index = np.random.choice(range(data.shape[0]),size = data.shape[0])   
        res = np.vstack((res,data[index]))
    print res.shape
    return res[:,2:],res[:,1:2]

X,Y = augment_data(data,n_boot = 3)
