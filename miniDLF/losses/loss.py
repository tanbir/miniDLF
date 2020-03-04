import numpy as np
from .cce import CCE
from .bce import BCE
from .mse import MSE

class Loss(object):
    def __init__(self, type='mse'):
        if type == 'mean_squared_error' or type == 'mse':            
            self.obj = MSE()
        elif type == 'categorical_crossentropy' or type == 'cce':            
            self.obj = CCE()
        elif type == 'binary_crossentropy' or type == 'bce':            
            self.obj = BCE()
        else:
            self.obj = MSE()
            
    def loss(self, a, y): 
        return self.obj.loss(a, y)
    
    def grad(self, a, y): 
        return self.obj.grad(a, y)
    