from .base import Dataset
import gzip
import pickle
import numpy as np

class MNIST(Dataset):
    def __init__(self, path, input_shape, n_classes):
        self.path = path  
        self.input_shape = input_shape
        self.n_classes = n_classes
        
        with gzip.open(self.path, 'rb') as f:
            train, validation, test = pickle.load(f, encoding='latin1')  
            t0 = np.array(train[0])
            v0 = np.array(validation[0])            
            c0 = np.concatenate((t0, v0), axis=0)
            t1 = np.array([self.__onehot__(x, n_classes) for x in train[1]])
            v1 = np.array([self.__onehot__(x, n_classes) for x in validation[1]])            
            c1 = np.concatenate((t1, v1), axis=0)
            train = c0, c1
                
            test = np.array(test[0]), np.array([self.__onehot__(x, n_classes) for x in test[1]])    
                
            Dataset.__init__(self, train, None, test, self.input_shape)
            
            tm = np.mean(self.train_x)
            self.train_x = np.array(self.train_x) - tm
            if self.validation_available == True:
                self.validation_x = np.array(self.validation_x) -tm
            if self.test_available == True:        
                self.test_x = np.array(self.test_x) - tm
                
   