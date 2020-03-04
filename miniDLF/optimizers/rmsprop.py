import numpy as np

################################################################################################################ 
class RMSProp(object):
    def __init__(self, lr=0.001, decay=1e-6, rho=0.9, epsilon=1e-6):
        self.lr = lr
        self.rho = rho
        self.decay = decay
        self.epsilon = epsilon
        self.a = None      # Running average of the squared gradients
        self.t = 0         # Timestamp
        
    def update(self, params, grads):                
        learning_rate = self.lr / (1.0 + self.decay * self.t)
        if self.a == None:
            # Initialize
            self.a = {}
            for y in params:
                self.a[y] = np.zeros_like(params[y])
                
        for y in params:
            # Update running average of the squared gradients
            self.a[y] = self.rho * self.a[y] + (1-self.rho) * (grads[y] ** 2)
            
            # Compute adaptive learning rate
            adaptive_lr = learning_rate / (np.sqrt(self.a[y])+self.epsilon)
            
            # Update parameters
            params[y] -= adaptive_lr * grads[y]
           
        self.t += 1
################################################################################################################