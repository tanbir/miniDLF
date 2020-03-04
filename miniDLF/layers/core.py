import numpy as np
from .base import Layer
from .activations import ReLU, ELU, Sigmoid, Tanh, Softmax

class Dense(Layer):
    def __init__(self, size, input_shape=None, trainable=True):
        Layer.__init__(self, trainable)
        self.size = size
        self.name = 'Dense     '
        self.X = None 
        
        self.input_shape = input_shape
        self.output_shape = (size, )
        
        # parameters
        self.params = {'W': None,'b': None}
        
        # optimizer
        self.optimizer = None
        
        # gradients
        self.grads = {'W':None,'b': None}
    
    def forward(self, X, train=True):
        self.X = X
        self.z = X @ self.params['W'] + self.params['b']
        return self.z
    
    def backward(self, delta):
        self.grads['W'] = self.X.T @ delta
        self.grads['b'] = np.sum(delta, axis=0)
        dX = delta @ self.params['W'].T 
        return dX, self.grads['b'], self.grads['W']
    
    def get_output_shape(self):
        return (self.size, )
    
    def initialize_params(self):
        self.params['W'] = np.random.randn(self.input_shape[0], self.size) / np.sqrt(self.size / 2.0)
        self.params['b'] = np.random.randn(1, self.size)  
        
    def set_optimizer(self, opt):
        self.optimizer = opt
        
    def update_params(self):
        self.optimizer.update(self.params, self.grads)
        
        

class Dropout(Layer):
    def __init__(self, p_dropout):
        Layer.__init__(self, False)
        self.shape = None
        self.u = None
        self.p = 1.0-p_dropout
        self.name = 'Dropout   '
        
        self.input_shape = None
        self.output_shape = None
        
    def forward(self, X, train=True): 
        self.shape = X.shape
        if train:
            self.u = np.random.binomial(1, self.p, size=X.shape) / self.p
            self.z = X * self.u 
        else:
            self.z = X
        return self.z

    def backward(self, delta):
        dX = self.u * delta
        return dX, None, None       
    
    def get_output_shape(self):
        return self.input_shape

class Flatten(Layer):
    def __init__(self):
        Layer.__init__(self, False)
        self.name = 'Flatten   '
        
        self.input_shape = None
        self.output_shape = None
        
    def forward(self, X, train=True):
        self.shape = X.shape
        self.z = X.reshape(X.shape[0], -1)
        return self.z

    def backward(self, delta):        
        dX = delta.reshape(self.shape)
        return dX, None, None
    
    def get_output_shape(self):
        return (np.prod(self.input_shape), )

class Reshape(Layer):
    def __init__(self, shape):
        Layer.__init__(self, False)        
        self.new_shape = shape
        self.name = 'Reshape   '
        
        self.input_shape = None
        self.output_shape = shape
    
    def forward(self, X):
        self.shape = X.shape        
        self.z = X.reshape( (X.shape[0],) + self.new_shape )
        return self.z
    
    def backward(self, delta):
        dX = delta.reshape(self.shape)    
        return dX, None, None
    
    def get_output_shape(self):
        return self.output_shape
    
class UpSampling2D(Layer):
    def __init__(self, size=(2,2)):
        Layer.__init__(self, False)        
        self.input_shape = None        
        self.size = size
        self.name = 'UpSampling2D'
        
        self.input_shape = None
        self.output_shape = None
 
    def forward(self, X, train=True):        
        self.shape = X.shape
        self.z = X.repeat(self.size[0], axis=2).repeat(self.size[1], axis=3)
        return self.z
    
    def backward(self, delta):        
        dX = delta[:, :, ::self.size[0], ::self.size[1]]
        return dX, None, None
    
    def get_output_shape(self):        
        C, H, W = self.input_shape
        return (C, H * self.size[0], W * self.size[1])
        
    
class Activation(Layer):
    count = 0
    def __init__(self, activation, alpha=0.0001, max_value=2.5):
        Layer.__init__(self, False)
        self.type = activation
        if activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'softmax':
            self.activation = Softmax()        
        elif activation == 'tanh':
            self.activation = Tanh()
        elif activation == 'relu':
            self.activation = ReLU(0.0, np.Infinity)
        elif activation == 'leaky_relu':
            self.activation = ReLU(alpha, max_value)
        elif activation == 'elu':
            self.activation = ELU(alpha, max_value)      
        self.name = 'Activation'
        
        self.input_shape = None
        self.output_shape = None
    
    def forward(self, X):
        self.z = self.activation.forward(X)
        return self.z
    
    def backward(self, delta):
        return self.activation.backward(delta), None, None
    
    def get_output_shape(self):
        return self.input_shape
    
    def gradient(self, X):
        return self.activation.gradient(X)