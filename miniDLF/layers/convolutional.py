from .base import Layer
from .utils import ImgCol
import numpy as np

class Conv2D(Layer):
    def __init__(self, filters, kernel, stride=1, padding=0, input_shape=None, trainable=True):
        Layer.__init__(self, trainable)  
        self.filters = filters # Number of filters 
        self.filter_H = kernel[0]
        self.filter_W = kernel[1]
        self.filter_D = None   
        self.stride = stride   
        self.padding = padding 

        if input_shape != None:
            self.input_shape = input_shape
            self.output_shape = self.get_output_shape()
        else:
            self.input_shape = None
            self.output_shape = None

               
        self.name = 'Conv2D    '
        
        
        self.shape = None
        self.x_col = None
        self.W_col = None
        
        # parameters
        self.params = {'W': None,'b': None}
        
        # optimizer
        self.optimizer = None
        
        # gradients
        self.grads = {'W':None,'b': None}
        
        
        self.imgcol = ImgCol(self.filter_H, self.filter_W, self.stride, self.padding)
        
    def get_n_fields(self, in_size, f_size): 
        return int((in_size - f_size + 2 * self.padding) / self.stride + 1) 
    
    def get_W_dimensions(self, input_shape):
        iD, iH, iW = input_shape
        return (self.filters, self.filter_H * self.filter_W * iD)
    
    def get_X_dimensions(self, input_shape):
        iD, iH, iW = input_shape
        n_fields = self.get_n_fields(iH, self.filter_H) * self.get_n_fields(iW, self.filter_W)
        return (self.filter_H * self.filter_W * iD, n_fields)
    
    def get_output_dimensions(self, input_shape):
        iD, iH, iW = input_shape
        return (self.filters, self.get_n_fields(iH, self.filter_H), self.get_n_fields(iW, self.filter_W))
 
    def forward(self, X, Train=True):        
        self.shape = X.shape
        
        self.W_col = self.params['W'].reshape(self.filters, -1)
        self.x_col = self.imgcol.img2col(X)

        self.z = self.W_col @ self.x_col + self.params['b']
        
        T = list(self.output_shape)
        T.append(X.shape[0])
        self.z = self.z.reshape(T).transpose(3, 0, 1, 2)  
        
        return self.z

    def backward(self, dout):         
        self.grads['b'] = np.sum(dout, axis=(0, 2, 3))
        self.grads['b'] = self.grads['b'].reshape(self.filters, -1)

        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.filters, -1)
        self.grads['W'] = dout_reshaped @ self.x_col.T
        self.grads['W'] = self.grads['W'].reshape(self.params['W'].shape)

        dX_col = self.W_col.T @ dout_reshaped       
        dX = self.imgcol.col2img(dX_col, self.shape)   
        
        return dX, self.grads['b'], self.grads['W']
    
    def get_output_shape(self):
        iD, iH, iW = self.input_shape
        return (self.filters, self.get_n_fields(iH, self.filter_H), self.get_n_fields(iW, self.filter_W))

    def initialize_params(self):
        W_shape = self.get_W_dimensions(self.input_shape)                
        self.params['W'] = np.random.randn(W_shape[0], W_shape[1])  / np.sqrt(W_shape[1] / 2.0)
        self.params['b'] = np.random.randn(self.filters, 1)   
        
    def set_optimizer(self, opt):
        self.optimizer = opt
        
    def update_params(self):
        self.optimizer.update(self.params, self.grads)