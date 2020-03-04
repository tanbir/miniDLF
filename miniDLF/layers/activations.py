import numpy as np

class ReLU(object): # covers Leaky ReLU and Clipped ReLU
    def __init__(self, alpha, max_value):
        self.z = None
        self.X = None
        self.alpha = alpha
        self.max_value = max_value
        
    def forward(self, X):  
        self.X = X
        self.z = np.maximum(self.alpha * X, X)
        if self.max_value != None:
            self.z[self.z >= self.max_value] = self.max_value
        return self.z
    
    def backward(self, delta):
        T = np.ones_like(self.X)
        T[self.X <= 0.0] = self.alpha
        T[self.z >= self.max_value] = 0.0
        dX = T * delta
        return dX  
    
    def gradient(self, X):        
        T = np.ones_like(X)
        T[X <= 0.0] = self.alpha
        T[X >= self.max_value] = 0.0
        return T

    
class Sigmoid(object):
    def __init__(self):
        self.z = None
        
    def forward(self, X):                
        self.z = 1.0 / (1 + np.exp(-X))        
        return self.z
    
    def backward(self, delta):        
        dX =  self.z * (1 - self.z) * delta
        return dX
    
    def gradient(self, X):
        z = 1.0 / (1 + np.exp(-X))
        return z * (1.0 - z)
    
class Tanh(object):
    def __init__(self):
        self.z = None
        
    def forward(self, X):    
        exps = np.exp(-2*X)
        self.z = (1.0 - exps)/(1.0 + exps)
        return self.z
    
    def backward(self, delta):
        dX = (1.0 - self.z**2) * delta
        return dX
    
    def gradient(self, X):
        exps = np.exp(-2*X)
        z = (1.0 - exps)/(1.0 + exps)        
        return 1.0 - z**2
    
class Softmax(object):
    def __init__(self):
        self.z = None
        
    def forward(self, X):
        e_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
        self.z = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return self.z

    def backward(self, delta):              
        return self.z * (1 - self.z) * delta 
    
    def gradient(self, X):
        e_x = np.exp(X - np.max(X, axis=-1, keepdims=True))
        z = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return z * (1 - z)
    
class ELU(object): # Covers SELU (Scaled Exponential Linear Units)
    def __init__(self, alpha, max_value):
        self.z = None
        self.X = None
        self.alpha = alpha        
        self.max_value = max_value
        
    def forward(self, X):  
        self.X = X
        self.ex = np.exp(X)
        self.z = (X > 0) * X + (X <= 0) * self.alpha * (self.ex - 1.0)        
        self.z[self.z >= self.max_value] = self.max_value
        return self.z
    
    def backward(self, delta):     
        T = self.alpha * self.ex
        T[self.X > 0.0] = 1     
        T[self.z >= self.max_value] = 0.0
        dX = T * delta
        return dX    