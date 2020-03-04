import numpy as np

################################################################################################################
class Adagrad(object):
    def __init__(self, lr=0.01, decay=1e-6, epsilon=1e-7):
        self.lr = lr        
        self.decay = decay
        self.epsilon = epsilon
        self.a = None       # Sum of squares of the gradients
        self.t = 0          # Timestamp  
        
    def update(self, params, grads):                
        learning_rate = self.lr / (1.0 + self.decay * self.t)
        if self.a == None:
            # Initialize
            self.a = {}
            for y in params:
                self.a[y] = np.zeros_like(params[y])
        
        for y in params:
            # Update sum of squares of the gradients
            self.a[y] += (grads[y] ** 2)
            
            # Compute adaptive learning rate
            adaptive_lr = learning_rate / (np.sqrt(self.a[y]+self.epsilon))
            
            # Update parameters
            params[y] -= adaptive_lr * grads[y]
        
        self.t += 1   
################################################################################################################