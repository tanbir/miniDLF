import numpy as np

################################################################################################################
class Nadam(object):
    def __init__(self, lr=0.001, decay=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-6):
        self.lr = lr
        self.beta_1 = beta_1   # Exponential decay-rate for the first moment estimate
        self.beta_2 = beta_2   # Exponential decay-rate for the second raw moment estimate
        self.decay = decay
        self.epsilon = epsilon
        self.m = None          # Running average of the first moment 
        self.v = None          # Running average of the second raw moment
        self.t = 0             # Timestamp
        
    def update(self, params, grads):                
        learning_rate = self.lr / (1.0 + self.decay * self.t)
        if self.m == None: 
            # Initialize
            self.m = {}
            self.v = {}
            for y in params:
                self.m[y] = np.zeros_like(params[y])
                self.v[y] = np.zeros_like(params[y])
                
        # Update learning rate considering bias-corrections for the biased moment estimates        
        lr_t = learning_rate * (np.sqrt(1.0 - np.power(self.beta_2, self.t + 1)) / 
                (1.0 - np.power(self.beta_1, self.t + 1)))
            
        tm = {}
        tv = {}
        for y in params:
            # Update biased first moment estimate
            self.m[y] = self.beta_1 * self.m[y] + (1-self.beta_1) * (grads[y])
            
            # Update biased second raw moment estimate
            self.v[y] = self.beta_2 * self.v[y] + (1-self.beta_2) * (grads[y] ** 2)
            
            # Move a bit further
            tm[y] = self.beta_1 * self.m[y] + (1-self.beta_1) * grads[y]
            tv[y] = self.beta_2 * self.v[y] + (1-self.beta_2) * (grads[y] ** 2)
            
            # Update parameters considering bias-corrected moment estimates
            params[y] -= lr_t * (tm[y]/ (np.sqrt(tv[y])+self.epsilon))
                
        self.t += 1             
################################################################################################################   