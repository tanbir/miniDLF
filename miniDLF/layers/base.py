import numpy as np

class Layer(object):
    def __init__(self, trainable = True):
        self.z = None
        
        self.name = None
        
        self.trainable = trainable
        self.n_trainable = 0
        
        self.input_shape = None
        self.output_shape = None
        
      