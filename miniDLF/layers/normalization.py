import numpy as np
from .base import Layer

class BatchNormalization(Layer):
    def __init__(self, momentum=.9, scale=True, center=True):
        Layer.__init__(self, True)
        self.momentum = momentum
        self.current_mean = None
        self.current_var = None
        self.scale = scale
        self.center = center
        
        self.mean = None
        self.var = None
        self.X_norm = None
        
        self.name = 'BatchNorm '
        
        self.input_shape = None
        self.output_shape = None
        
        # parameters
        self.params = {'gamma': None,'beta': None}
        
        # optimizer
        self.optimizer = None
        
        # gradients
        self.grads = {'gamma':None,'beta': None}
        

    def forward(self, X, train=True):
        self.X = X
        if train:
            self.mean = np.mean(self.X, axis=0)
            self.var = np.var(self.X, axis=0)

            self.X_norm = (X - self.mean) / np.sqrt(self.var + 1e-8)            
            self.z = self.params['gamma'] * self.X_norm + self.params['beta']
                       
            self.current_mean = self.momentum * self.current_mean + (1.0 - self.momentum) * self.mean
            self.current_var = self.momentum * self.current_var + (1.0 - self.momentum) * self.var
        else:
            X_norm = (X - self.current_mean) / np.sqrt(self.current_var + 1e-8)
            self.z = self.params['gamma'] * X_norm + self.params['beta']
        return self.z
    
    def backward(self, delta):
        self.grads['beta'] = np.sum(delta, axis=0)
        self.grads['gamma'] = np.sum(delta * self.X_norm, axis=0)
        
        N = self.X.shape[0]
        X_mean = self.X - self.mean
        stdev_inv = 1.0 / np.sqrt(self.var + 1e-8)        
        dX_norm = self.params['gamma'] * delta
        dvar = np.sum(dX_norm * X_mean, axis=0) * -0.5 * stdev_inv**3
        dmean = np.sum(dX_norm * -stdev_inv, axis=0) + dvar * np.mean(-2.0 * X_mean, axis=0)
        
        dX = (dX_norm * stdev_inv) + (dvar * 2 * X_mean / N) + (dmean / N)
        
        return dX, self.grads['beta'], self.grads['gamma']
    
    def get_output_shape(self):
        return self.input_shape
    
    def initialize_params(self):
        self.current_mean = np.zeros(self.input_shape)
        self.current_var = np.zeros(self.input_shape)
        self.params['gamma'] = np.ones(self.input_shape)
        self.params['beta'] = np.zeros(self.input_shape)
        
    def set_optimizer(self, opt):
        self.optimizer = opt
        
    def update_params(self):
        self.optimizer.update(self.params, self.grads)