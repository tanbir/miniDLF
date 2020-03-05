import pickle
import gzip
import random
import numpy as np
import sys
import time

class Dataset(object):
    def __init__(self, train, validation, test, input_shape=None):
        self.train_x = train[0]         
        self.train_y = train[1]                        
        self.n_train = len(train[0])
        if validation != None:
            self.validation_available = True
            self.validation_x = validation[0]
            self.validation_y = validation[1]
            self.n_validation = len(validation[0])
        else:
            self.validation_available = False
        if test != None:
            self.test_available = True
            self.test_x = test[0]
            self.test_y = test[1]
            self.n_test = len(test[0])
        else:
            self.test_available = False
        if input_shape != None:
            self.set_input_shape(input_shape)
            
    def set_input_shape(self, input_shape):
        self.train_x = [np.reshape(X, input_shape) for X in self.train_x]
        if self.validation_available == True:
            self.validation_x = [np.reshape(X, input_shape) for X in self.validation_x]
        self.test_x = [np.reshape(X, input_shape) for X in self.test_x]
    
    def shuffle_train(self):
        idx = np.arange(self.n_train)
        np.random.shuffle(idx)
        self.train_x = np.array(self.train_x)[idx]
        self.train_y = np.array(self.train_y)[idx]     
      
    def get_batch(self, _type, ind, batch_size):
        if _type == 'train':            
            mini_x = self.train_x[ind * batch_size : (ind+1) * batch_size]
            mini_y = self.train_y[ind * batch_size : (ind+1) * batch_size]
        elif _type=='validation':
            mini_x = self.validation_x[ind * batch_size : (ind+1) * batch_size]
            mini_y = self.validation_y[ind * batch_size : (ind+1) * batch_size]
        elif _type=='test':
            mini_x = self.test_x[ind * batch_size : (ind+1) * batch_size]
            mini_y = self.test_y[ind * batch_size : (ind+1) * batch_size]
        return mini_x, mini_y

    def shuffle_data(self, X, y, seed=None):
        if seed:
            np.random.seed(seed)
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        return X[idx], y[idx]    
    
    def train_test_split(self, X, y, test_size=0.5, shuffle=True, seed=None):
        if shuffle:
            X, y = self.shuffle_data(X, y, seed)
        split_i = len(y) - int(len(y) * test_size)
        X_train, X_test = X[:split_i], X[split_i:]
        y_train, y_test = y[:split_i], y[split_i:]

        return X_train, X_test, y_train, y_test

    def __onehot__(self, v, n_classes):
        y = np.zeros(n_classes)
        y[v] = 1
        return y   
