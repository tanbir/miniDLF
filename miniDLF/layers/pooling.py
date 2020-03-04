import numpy as np
from .base import Layer
from .utils import ImgCol

class Pooling2D(Layer):
    def __init__(self, size=2, stride=2, mode='max'):  
        Layer.__init__(self, False)  
        self.size = size
        self.stride = stride
        self.mode = mode
        self.flatten = False
        self.imgcol = ImgCol(self.size, self.size, self.stride, 0) 
        self.max_idx = None
        
        self.shape = None
        self.x_col = None
        
        self.name = 'Pooling   '      
        
        self.input_shape = None
        self.output_shape = None
        
    def get_n_fields(self, in_size): 
        return int((in_size - self.size) / self.stride + 1)    
        
    def get_output_dimensions(self, input_shape):
        iD, iH, iW = input_shape
        return (iD, self.get_n_fields(iH), self.get_n_fields(iW))
    
    def get_output_shape(self):
        iD, iH, iW = self.input_shape
        return (iD, self.get_n_fields(iH), self.get_n_fields(iW))
    
    def forward(self, X):
        self.shape = X.shape
        N, iD, iH, iW = X.shape
        
        oD, oH, oW = self.output_shape
        X_reshaped = X.reshape(N * iD, 1, iH, iW)
        self.x_col = self.imgcol.img2col(X_reshaped)
        
        if self.mode == 'max':
            self.max_idx = np.argmax(self.x_col, axis=0)
            out = self.x_col[self.max_idx, range(self.max_idx.size)]
        elif self.mode == 'avg':
            out = np.mean(self.x_col, axis=0)
       
        self.z = out.reshape(oH, oW, N, oD).transpose(2, 3, 0, 1)
        return self.z
    
    
    def backward(self, dout):       
        N, iD, iH, iW = self.shape
        
        dX_col = np.zeros_like(self.x_col)
        
        dout_col = dout.transpose(2, 3, 0, 1).ravel()
        
        if self.mode == 'max':        
            dX = dX_col[self.max_idx, range(dout_col.size)] = dout_col
        elif self.mode == 'avg':
            dX = dX_col[:, range(dout_col.size)] = 1. / dX_col.shape[0] * dout_col    

        dX = self.imgcol.col2img(dX_col, (N * iD, 1, iH, iW))
        dX = dX.reshape(self.shape)
        return dX, None, None        