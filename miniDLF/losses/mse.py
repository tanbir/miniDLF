import numpy as np

class MSE(object):
    def __init__(self):
        pass
    
    def loss(self, p, y):
        m = p.shape[0]                
        data_loss = 0.5 * np.sum((p - y)**2)     
        
        return data_loss / m
    
    def grad(self, p, y):
        g = p - y
        return g
