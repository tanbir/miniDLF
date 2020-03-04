import numpy as np

class BCE(object):
    def __init__(self): 
        pass
    
    def __onehot__(self, v, n_classes):
        y = np.zeros(n_classes)
        y[v] = 1
        return y
    
    def loss(self, p, y):     
        # y: for each input in the batch, the known classification probability             
        # p: for each input in the batch, the predicted classification probability
        
        p = np.clip(p, 1e-15, 1-1e-15)   
        m = p.shape[0]
        y_dims = len(y.shape) 
        if y_dims == 1:                # y shape: (batch_size, )  
            n = p.shape[-1]
            y_ = np.array([self.__onehot__(v, n) for v in y])
            nlog_sum = np.sum(- y_ * np.log(p) - (1 - y_) * np.log(1 - p), axis=-1)            
            return np.mean(nlog_sum)
        
        elif y_dims == 2:                # shape: (batch_size, n_classes) 
            nlog_sum = np.sum(- y * np.log(p) - (1 - y) * np.log(1 - p), axis=-1)            
            return np.mean(nlog_sum)
        
        elif y_dims == 3:              # y shape: (batch_size, timesteps, input_dims)  
            nlog_sum = np.sum(np.sum(- y * np.log(p) - (1 - y) * np.log(1 - p), axis=-1), axis=-2)            
            return np.mean(nlog_sum)    
    
    def grad(self, p, y):        
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return (- (y / p) + (1 - y) / (1 - p))       
        
 