import numpy as np

################################################################################################################
class SGD(object):
    def __init__(self, lr=0.01, decay=1e-6, momentum=0.9, nesterov=True):
        self.lr = lr
        self.decay = decay
        self.m = momentum
        self.nesterov = nesterov
        self.v = None
        self.t = 0
    
    def update(self, params, grads):                    
        learning_rate = self.lr / (1.0 + self.decay * self.t)
        if self.v == None:
            self.v = {}
            for y in params:
                self.v[y] = np.zeros_like(params[y])
             
        if self.nesterov == True:
            for y in params:
                self.v[y] = self.m * self.v[y] + (1.0 - self.m) * grads[y]
                params[y] -= learning_rate * (self.m * self.v[y]  + (1.0 - self.m) * grads[y])
        else:
            for y in params:
                self.v[y] = self.m * self.v[y] + (1.0 - self.m) * grads[y]
                params[y] -= learning_rate * self.v[y]

        self.t += 1
################################################################################################################       