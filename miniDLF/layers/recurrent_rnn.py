from .base import Layer
from .activations import Sigmoid, Tanh
import numpy as np
import copy


class RNN(Layer):
    def __init__(self, n_units, input_shape=None):
        super().__init__(True)
        self.n_units = n_units
        self.input_shape = input_shape
        self.tanh = Tanh()
        self.name = "RNN       "
        
        # Parameters
        self.params = {'U':None, 
                       'W':None, 
                       'V':None,
                       'bw': None,
                       'bv': None}
        
        # Optimizers
        self.optimizer = None
        
        self.grads = {'U':None, 
                      'W':None, 
                      'V':None,
                      'bw': None,
                      'bv': None}
        

    def initialize_params(self):       
        timesteps, input_dim = self.input_shape
        
        # Initialize the weights
        self.params['U'] = np.random.randn(input_dim, self.n_units) / np.sqrt(self.n_units)
        self.params['W'] = np.random.randn(self.n_units, self.n_units) / np.sqrt(self.n_units)
        self.params['V'] = np.random.randn(self.n_units, input_dim) / np.sqrt(input_dim)
        
        # Initialize the biases
        self.params['bw'] = np.zeros((1, self.n_units))
        self.params['bv'] = np.zeros((1, input_dim))

    def set_optimizer(self, opt):
        # Initialize optimizer
        self.optimizer = opt


    def forward(self, X, train=True):
        batch_size, timesteps, input_dim = np.array(X).shape
        
        self.X = X
        
        self.h = np.zeros((batch_size, timesteps+1, self.n_units))  # states
        self.a = np.zeros((batch_size, timesteps, self.n_units))    # inputs
        self.y = np.zeros((batch_size, timesteps, input_dim))       # outputs    
        
        self.h[:, -1] = np.zeros((batch_size, self.n_units))
        for t in range(timesteps):            
            self.a[:, t] = (X[:, t] @ self.params['U'] + self.h[:, t-1] @ self.params['W'] + self.params['bw'])                        
            self.h[:, t] = self.tanh.forward(self.a[:, t])
            self.y[:, t] = self.h[:, t] @ self.params['V'] + self.params['bv']           
        return self.y    
    
    def backward(self, delta):
        _, timesteps, _ = delta.shape

        # Initialize placeholderfor gradients w.r.t each parameter
        self.grads['U']  = np.zeros_like(self.params['U'])
        self.grads['W']  = np.zeros_like(self.params['W'])
        self.grads['V']  = np.zeros_like(self.params['V'])
        
        self.grads['bw']  = np.zeros_like(self.params['bw'])
        self.grads['bv']  = np.zeros_like(self.params['bv'])
        
        # Initialize gradient w.r.t layer input X
        dX = np.zeros_like(delta)

        # Back Propagation Through Time
        for t in reversed(range(timesteps)):                    
            dy_t = delta[:, t]
            
            self.grads['V'] += self.h[:, t].T @ dy_t   
            self.grads['bv'] += np.sum(dy_t, axis=0)
            
            dh_t = dy_t @ self.params['V'].T
            
            dt = dh_t * (1.0 - self.h[:, t]**2)
            
            self.grads['W'] += self.h[:, t-1].T @ dt            
            self.grads['U'] += self.X[:, t].T @ dt
            
            self.grads['bw'] += np.sum(dt, axis=0)
            
            dX[:, t] = dt @ self.params['U'].T            
        return dX, None, None

    def update_params(self):
        self.optimizer.update(self.params, self.grads)
    
    def get_output_shape(self):
        return self.input_shape
