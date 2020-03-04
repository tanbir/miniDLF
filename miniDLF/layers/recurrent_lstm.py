from .base import Layer
from .activations import Sigmoid, Tanh
import numpy as np
import copy
import random

class LSTM(Layer):
    def __init__(self, n_units, input_shape=None):        
        super().__init__(True)
        self.n_units = n_units
        self.input_shape = input_shape
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()        
        self.name = "LSTM      "
        
        # Parameters
        self.params = {'Wxf':None,                        
                       'Whf':None,
                       'bf':None,
                       'Wxi':None, 
                       'Whi':None,
                       'bi':None,
                       'Wxo':None,                                                                     
                       'Who':None,
                       'bo':None,
                       'Wxg':None,                                                                     
                       'Whg':None,
                       'bg':None,
                       'Why':None,
                       'by':None}
        
        # Optimizers
        self.optimizer = None
        
        # Gradients
        self.grads = {'Wxf':None,                        
                      'Whf':None,
                      'bf':None,
                      'Wxi':None, 
                      'Whi':None,
                      'bi':None,
                      'Wxo':None,                                                                     
                      'Who':None,
                      'bo':None,
                      'Wxg':None,                                                                     
                      'Whg':None,
                      'bg':None,
                      'Why':None,
                      'by':None}
              
    def initialize_params(self):   
        D = self.input_shape[1]
        H = self.n_units
        # Initialize the weights                
        self.params['Whf'] = np.random.randn(H, H) / np.sqrt(H / 2.0)
        self.params['Whi'] = np.random.randn(H, H) / np.sqrt(H / 2.0)
        self.params['Who'] = np.random.randn(H, H) / np.sqrt(H / 2.0)
        self.params['Whg'] = np.random.randn(H, H) / np.sqrt(H / 2.0)  
        self.params['Wxf'] = np.random.randn(D, H) / np.sqrt(D / 2.0)
        self.params['Wxi'] = np.random.randn(D, H) / np.sqrt(D / 2.0)
        self.params['Wxo'] = np.random.randn(D, H) / np.sqrt(D / 2.0)
        self.params['Wxg'] = np.random.randn(D, H) / np.sqrt(D / 2.0)                
        self.params['Why'] = np.random.randn(H, D) / np.sqrt(H / 2.0)                
        
        # Initialize the biases
        self.params['bf'] = np.zeros((1, H))
        self.params['bi'] = np.zeros((1, H))
        self.params['bo'] = np.zeros((1, H))
        self.params['bg'] = np.zeros((1, H))                
        self.params['by'] = np.zeros((1, D))    
        
        self.dh_next = np.zeros((1, H))
        self.dc_next = np.zeros((1, H))
    
    def set_optimizer(self, opt):
        # Initialize optimizer
        self.optimizer = opt
        
    def forward(self, X, train=True):
        N, T, D = np.array(X).shape
        #batch_size, timesteps, input_dim = np.array(X).shape
        self.timesteps = T               
        H = self.n_units
        
        self.f = np.zeros((N, T, H))  # forget gate outputs
        self.i = np.zeros((N, T, H))  # input gate outputs        
        self.o = np.zeros((N, T, H))  # output gate outputs
        self.g = np.zeros((N, T, H))  # input modulation gate outputs
        
        
        self.c = np.zeros((N, T+1, H)) # cell states                
        self.h = np.zeros((N, T+1, H)) # hidden states      
        
        self.x = np.zeros((N, T, D))      # outputs    
        self.y = np.zeros((N, T, D))      # outputs    
        
        self.h[:, -1] = np.zeros((N, H))
        self.c[:, -1] = np.zeros((N, H))
        for t in range(T):            
            self.x[:, t] = X[:, t]
            
            # Forget gate output shape: (batch_size, self.n_units) 
            self.f[:, t] = X[:, t] @ self.params['Wxf'] + self.h[:, t-1] @ self.params['Whf'] + self.params['bf']
            self.f[:, t] = self.sigmoid.forward(self.f[:, t])
                                   
            # Input gate output shape: (batch_size, self.n_units) 
            self.i[:, t] = X[:, t] @ self.params['Wxi'] + self.h[:, t-1] @ self.params['Whi'] + self.params['bi']
            self.i[:, t] = self.sigmoid.forward(self.i[:, t])            
            
            # Output gate output shape: (batch_size, self.n_units) 
            self.o[:, t] = X[:, t] @ self.params['Wxo'] + self.h[:, t-1] @ self.params['Who'] + self.params['bo']
            self.o[:, t] = self.sigmoid.forward(self.o[:, t])            
            
            # Input modulation gate output shape: (batch_size, self.n_units) 
            self.g[:, t] = X[:, t] @ self.params['Wxg'] + self.h[:, t-1] @ self.params['Whg'] + self.params['bg']
            self.g[:, t] = self.tanh.forward(self.g[:, t])
                
            # At timestep t: Cell memory state with shape (batch_size, self.n_units)
            self.c[:, t] = self.c[:, t-1] * self.f[:, t] + self.i[:, t] * self.g[:, t]            
            
            # At timestep t, hidden states with shape: (batch_size, self.n_units)
            self.h[:, t] = self.o[:, t] * self.tanh.forward(self.c[:, t])               
            
            # At timestep t: Cell output with shape (batch_size, input_dim)                        
            self.y[:, t] = self.h[:, t] @ self.params['Why'] + self.params['by']
        return self.y
      
    def backward(self, delta):
        # Initialize placeholder for gradients w.r.t each parameter
        self.grads['Wxf']  = np.zeros_like(self.params['Wxf'])
        self.grads['Wxi']  = np.zeros_like(self.params['Wxi'])
        self.grads['Wxo']  = np.zeros_like(self.params['Wxo'])
        self.grads['Wxg']  = np.zeros_like(self.params['Wxg'])
        
        self.grads['Whf']  = np.zeros_like(self.params['Whf'])
        self.grads['Whi']  = np.zeros_like(self.params['Whi'])
        self.grads['Who']  = np.zeros_like(self.params['Who'])
        self.grads['Whg']  = np.zeros_like(self.params['Whg'])
        
        self.grads['Why']  = np.zeros_like(self.params['Why'])
        
        self.grads['bf']  = np.zeros_like(self.params['bf'])
        self.grads['bi']  = np.zeros_like(self.params['bi'])
        self.grads['bo']  = np.zeros_like(self.params['bo'])
        self.grads['bg']  = np.zeros_like(self.params['bg'])
        self.grads['by']  = np.zeros_like(self.params['by'])
        
        # Initialize gradient w.r.t layer input X
        dX = np.zeros_like(self.x)
                
        # Back Propagation Through Time
        for t in reversed(range(self.timesteps)):                     
            ctanh = self.tanh.forward(self.c[:, t])
            dy_t = delta[:, t]

            dh_t = dy_t @ self.params['Why'].T + self.dh_next * 0.01          
            
            dc_t = dh_t * self.o[:, t] * (1.0 - ctanh**2) + self.dc_next            
            
            do_t = dh_t * ctanh * (self.o[:, t] * (1.0 - self.o[:, t]))                                     
            df_t = dc_t * self.c[:, t-1] * (self.f[:, t] * (1.0 - self.f[:, t]))                        
            di_t = dc_t * self.g[:, t] * (self.i[:, t] * (1.0 - self.i[:, t]))                     
            dg_t = dc_t * self.i[:, t] * (1.0 - self.g[:, t]**2)           
                        
               
                
            self.grads['Why'] += self.h[:, t].T @ dy_t    
            self.grads['by'] += np.sum(dy_t, axis=0)
            
            self.grads['Wxf'] += self.x[:, t].T @ df_t
            self.grads['Wxi'] += self.x[:, t].T @ di_t
            self.grads['Wxo'] += self.x[:, t].T @ do_t
            self.grads['Wxg'] += self.x[:, t].T @ dg_t
            
            self.grads['Whf'] += self.h[:, t-1].T @ df_t
            self.grads['Whi'] += self.h[:, t-1].T @ di_t
            self.grads['Who'] += self.h[:, t-1].T @ do_t
            self.grads['Whg'] += self.h[:, t-1].T @ dg_t
            
            self.grads['bf'] += np.sum(df_t, axis=0)
            self.grads['bi'] += np.sum(di_t, axis=0)
            self.grads['bo'] += np.sum(do_t, axis=0)
            self.grads['bg'] += np.sum(dg_t, axis=0)
            
            
            
            
            # shape: (batch_size, input_dim)
            dXf = df_t @ self.params['Wxf'].T
            dXi = di_t @ self.params['Wxi'].T
            dXg = dg_t @ self.params['Wxg'].T
            dXo = do_t @ self.params['Wxo'].T 
            
            dX[:, t] = (dXf + dXi + dXg + dXo) 
            
            # shape: (batch_size, self.n_units)
            dHf = df_t @ self.params['Whf'].T
            dHi = di_t @ self.params['Whi'].T 
            dHg = dg_t @ self.params['Whg'].T 
            dHo = do_t @ self.params['Who'].T   
            
            self.dh_next = (dHo + dHf + dHi + dHg)            
                        
            self.dc_next = self.f[:, t] * dc_t                                      
        return dX, None, None
                
    def update_params(self):
        self.optimizer.update(self.params, self.grads)
    
    def get_output_shape(self):
        return self.input_shape    