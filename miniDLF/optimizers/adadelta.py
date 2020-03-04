import numpy as np

class Adadelta(object):
    def __init__(self, lr=0.001, decay=1e-6, rho=0.95, eps=1e-6):  
        self.lr = lr
        self.eps = eps
        self.rho = rho
        self.t = 0
        
        self.m_sg = None      # Running mean of squared gradients        
        self.m_su = None      # Running mean of squared updates 
        self.u = None         # Updates

    def update(self, params, grads):
        learning_rate = self.lr / (1.0 + self.decay * self.t)
        
        if self.m_sg == None:
            # Initialize running means and updates
            self.m_sg = {}
            self.m_su = {}
            self.u = {}
            for y in params:
                self.m_sg[y] = np.zeros_like(grads[y])
                self.m_su[y] = np.zeros_like(params[y])
                self.u[y] = np.zeros_like(params[y])
        
        for y in params:
            # Update running mean of squared gradients
            self.m_sg[y] = self.rho * self.m_sg[y] + (1 - self.rho) * (grads[y] ** 2) 
            
            # Compute adaptive learning rate
            rms_su = np.sqrt(self.m_su[y] + self.eps)
            rms_grad = np.sqrt(self.m_sg[y] + self.eps)
            adaptive_lr = learning_rate * rm_su / rms_grad
            
            # Compute update
            self.u[y] = adaptive_lr * grads[y]
            
            # Updtae running mean of squared updates
            self.m_su[y] = self.rho * self.m_su[y] + (1 - self.rho) * (self.u[y] ** 2)
            
            # Update parameters
            params[y] -= self.u[y]
        
        self.t += 1
