from .base import Layer
from .activations import Sigmoid, Tanh
import numpy as np
import copy

class GRU(Layer):
    def __init__(self, n_units, input_shape=None):        
        super().__init__(True)                
        self.n_units = n_units
        self.input_shape = input_shape
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()    
        self.name = "GRU       "
        
        # Parameters
        self.params = {'Wxu':None, 
                       'Whu':None,
                       'bu':None,                       
                       'Wxr':None,                                                                     
                       'Whr':None,
                       'br':None,
                       'Wxc':None,                                                                     
                       'Whc':None,
                       'bc':None,
                       'Why':None,
                       'by':None}
        
        # Optimizers
        self.optimizer = None
        
        # Gradients
        self.grads = {'Wxu':None, 
                      'Whu':None,
                      'bu':None,                      
                      'Wxr':None,                                                                     
                      'Whr':None,
                      'br':None,
                      'Wxc':None,                                                                     
                      'Whc':None,
                      'bc':None,
                      'Why':None,
                      'by':None}        
    
    def initialize_params(self):   
        input_dim = self.input_shape[1]
        # Initialize the weights                
        self.params['Whu'] = np.random.randn(self.n_units, self.n_units) / np.sqrt(self.n_units / 2.0)
        self.params['Whr'] = np.random.randn(self.n_units, self.n_units) / np.sqrt(self.n_units / 2.0)                
        self.params['Whc'] = np.random.randn(self.n_units, self.n_units) / np.sqrt(self.n_units / 2.0)                
        self.params['Wxu'] = np.random.randn(input_dim, self.n_units) / np.sqrt(input_dim / 2.0)
        self.params['Wxr'] = np.random.randn(input_dim, self.n_units) / np.sqrt(input_dim / 2.0)                
        self.params['Wxc'] = np.random.randn(input_dim, self.n_units) / np.sqrt(input_dim / 2.0)                
        
        self.params['Why'] = np.random.randn(self.n_units, input_dim) / np.sqrt(input_dim / 2.0)                
        
        # Initialize the biases
        self.params['bu'] = np.zeros((1, self.n_units))
        self.params['br'] = np.zeros((1, self.n_units))                
        self.params['bc'] = np.zeros((1, self.n_units))                
        self.params['by'] = np.zeros((1, input_dim))    
        
        self.dh_next = np.zeros((1, self.n_units))        
    
    def set_optimizer(self, opt):
        # Initialize optimizer
        self.optimizer = opt
        
    def forward(self, X, train=True):
        batch_size, timesteps, input_dim = np.array(X).shape
        self.timesteps = timesteps
                
        self.u = np.zeros((batch_size, timesteps, self.n_units))  # update gate outputs                
        self.r = np.zeros((batch_size, timesteps, self.n_units))  # relevance gate outputs
        
        
        self.c = np.zeros((batch_size, timesteps+1, self.n_units)) # cell states                
        self.h = np.zeros((batch_size, timesteps+1, self.n_units)) # hidden states      
        
        self.x = np.zeros((batch_size, timesteps, input_dim))      # inputs    
        self.y = np.zeros((batch_size, timesteps, input_dim))      # outputs    
        
        self.h[:, -1] = np.zeros((batch_size, self.n_units))
        self.c[:, -1] = np.zeros((batch_size, self.n_units))
        for t in range(timesteps):            
            self.x[:, t] = X[:, t]
                                               
            # Update gate output shape: (batch_size, self.n_units) 
            self.u[:, t] = X[:, t] @ self.params['Wxu'] + self.h[:, t-1] @ self.params['Whu'] + self.params['bu']
            self.u[:, t] = self.sigmoid.forward(self.u[:, t])            
            
            # Relevance gate output shape: (batch_size, self.n_units) 
            self.r[:, t] = X[:, t] @ self.params['Wxr'] + self.h[:, t-1] @ self.params['Whr'] + self.params['br']
            self.r[:, t] = self.sigmoid.forward(self.r[:, t])
                
            # At timestep t: Cell memory state with shape (batch_size, self.n_units)
            self.c[:, t] = X[:, t] @ self.params['Wxc'] 
            self.c[:, t] +=  (self.h[:, t-1] * self.r[:, t]) @ self.params['Whc'] 
            self.c[:, t] += self.params['bc']       
            
            # At timestep t, hidden states with shape: (batch_size, self.n_units)            
            self.h[:, t] = (1.0 - self.u[:, t]) * self.h[:, t-1] + self.u[:, t] * self.tanh.forward(self.c[:, t])              
            
            # At timestep t: Cell output with shape (batch_size, input_dim)                        
            self.y[:, t] = self.h[:, t] @ self.params['Why'] + self.params['by']
        
        return self.y
    
    def backward(self, delta):
        # Initialize placeholder for gradients w.r.t each parameter        
        self.grads['Wxu']  = np.zeros_like(self.params['Wxu'])
        self.grads['Wxr']  = np.zeros_like(self.params['Wxr'])
        self.grads['Wxc']  = np.zeros_like(self.params['Wxc'])
        
        self.grads['Whu']  = np.zeros_like(self.params['Whu'])
        self.grads['Whr']  = np.zeros_like(self.params['Whr'])
        self.grads['Whc']  = np.zeros_like(self.params['Whc'])
        
        self.grads['Why']  = np.zeros_like(self.params['Why'])
        
        self.grads['bu']  = np.zeros_like(self.params['bu'])
        self.grads['br']  = np.zeros_like(self.params['br'])
        self.grads['bc']  = np.zeros_like(self.params['bc'])
        self.grads['by']  = np.zeros_like(self.params['by'])
        
        # Initialize gradient w.r.t layer input X
        dX = np.zeros_like(self.x)
                
        # Back Propagation Through Time
        for t in reversed(range(self.timesteps)):                     
            ctanh = self.tanh.forward(self.c[:, t])
            dy_t = delta[:, t]
            
            
            ########################################################
            dh_t = dy_t @ self.params['Why'].T + self.dh_next * 0.01
            self.grads['Why'] += self.h[:, t].T @ dy_t
            self.grads['by'] += np.sum(dy_t, axis=0)
                 
            # tanh #################################################
            dt = dh_t * self.u[:, t] * (1.0 - ctanh**2)
            
            self.grads['Wxc'] += self.x[:, t].T @ dt
            self.grads['Whc'] += (self.r[:, t] * self.h[:, t-1]).T @ dt

            self.grads['bc'] += np.sum(dt, axis=0)
            
            dXc = dt @ self.params['Wxc'].T
            dHc = dt @ self.params['Whc'].T
            
            # Update gate ##########################################
            du_t = dh_t * (ctanh - self.h[:, t-1])
            dt = du_t * self.u[:, t] * (1.0 - self.u[:, t])
            
            self.grads['Wxu'] += self.x[:, t].T @ dt
            self.grads['Whu'] += self.h[:, t-1].T @ dt

            self.grads['bu'] += np.sum(dt, axis=0)            

            dXu = dt @ self.params['Wxu'].T
            dHu = dt @ self.params['Whu'].T

            
            
            # Reset gate ###########################################
            dr_t = dHc * self.h[:, t-1] 
            dt = dr_t * self.r[:, t] * (1.0 - self.r[:, t])
            
            self.grads['Wxr'] += self.x[:, t].T @ dt
            self.grads['Whr'] += self.h[:, t-1].T @ dt

            self.grads['br'] += np.sum(dt, axis=0)            

            dXr = dt @ self.params['Wxr'].T
            dHr = dt @ self.params['Whr'].T
            
            ########################################################
            
            dX[:, t] = (dXr + dXu)
            
            self.dh_next = ((1.0 - self.u[:, t]) * dh_t + self.r[:, t] * dHc + dHr + dHu)
        return dX, None, None
                
    def update_params(self):
        self.optimizer.update(self.params, self.grads)
    
    def get_output_shape(self):
        return self.input_shape    