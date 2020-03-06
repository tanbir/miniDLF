import numpy as np
import time
import copy
from miniDLF.layers import *
from miniDLF.optimizers import *
from miniDLF.losses import *
import tqdm

class Sequential(object):
    def __init__(self):
        self.layers = []
              
        self.nLayers = 0   
        self.loss = None
        self.loss_function = None
        self.optimizer = None
        self.minibatch_size = 10
        self.mode = None
        self.train_accuracies = []
        self.test_accuracies = []   
        self.regression = False
                
    def add(self, layer):
        self.layers.append(layer);     
        self.nLayers += 1
        
    def summary(self):
        total = 0
        for i in range(0, self.nLayers):
            n_trainable = 0
            if self.layers[i].trainable==True:
                for y in self.layers[i].params:
                    n_trainable += self.layers[i].params[y].size
            print(self.layers[i].name,': ', ' input_shape = ', 
                  self.layers[i].input_shape, ' output_shape = ', 
                  self.layers[i].output_shape, ' trainable parameters = ',n_trainable)  
            total += n_trainable
        print('Total # trainable parameters:', total)
        
    def get_optimizer(self, optimizer):
        # Set up optimizer
        if isinstance(optimizer, str):
            
            if optimizer.lower() == 'sgd': opt = SGD();
            elif optimizer.lower() == 'adagrad': opt = Adagrad();
            elif optimizer.lower() == 'rmsprop': opt = RMSprop();
            elif optimizer.lower() == 'adam': opt = Adam();                
            elif optimizer.lower() == 'nadam': opt = Nadam();                
            elif optimizer.lower() == 'amsgrad': opt = AMSGrad(); 
            else: opt = SGD();
        else:
            if (isinstance(optimizer, SGD) or 
                isinstance(optimizer, RMSProp) or 
                isinstance(optimizer, Adagrad) or 
                isinstance(optimizer, Adam) or
                isinstance(optimizer, Nadam) or
                isinstance(optimizer, AMSGrad)):               
                opt = optimizer
            else:
                opt = SGD();
        return copy.copy(opt)
    
    
    def compile(self, 
                loss='cce', 
                optimizer='sgd'):
        
        # Set loss function object
        self.loss_function = Loss(loss)
        
        if self.nLayers == 0:
            print('No layers added')
            sys.exit()
        elif loss == 'categorical_crossentropy' or loss == 'cce':
            if ((not isinstance(self.layers[-1], Activation)) or (self.layers[-1].type != 'softmax')):
                print(self.layers[-1].activation,'A softmax actiation must be used as the last layer')
                sys.exit()
        elif self.layers[0].input_shape==None:
            print('Input shape must be specified')
            sys.exit()
        else:
            pass
        
        input_shape = np.array(self.layers[0].input_shape)
        for i in range(0, self.nLayers):
            self.layers[i].input_shape = input_shape
            output_shape = self.layers[i].get_output_shape()
            self.layers[i].output_shape = output_shape 
            
            if self.layers[i].trainable == True:            
                self.layers[i].set_optimizer(self.get_optimizer(optimizer))
                self.layers[i].initialize_params()
            input_shape = output_shape
    
    def __feedforward__(self, X, Train=True):       
        a = X
        for i in range(0, self.nLayers): 
            a = self.layers[i].forward(a)            
        return a
    
    def __backpropagate__(self, X, Y):                
        delta = self.loss_function.grad(self.layers[-1].z, Y) 
        for i in range(1, self.nLayers+1):            
            delta, tb, tW = self.layers[-i].backward(delta)
            if self.layers[-i].trainable == True:
                self.layers[-i].update_params()
    
    def __update_mini_batch__(self, minibatch):
        X, y = minibatch
        X_ = np.array(X)
        y_pred = self.__feedforward__(X_)
        self.__backpropagate__(X_, y)   
        
        self.loss += self.loss_function.loss(y_pred, y) / self.num_train_minibatches

    def __progress__(self, i, n, t, c='='):
        A = [int((n*i)/t + 1) for i in range(t)]
        B = set(A)
        if i in B:
            print(c, end="")

        

    def fit(self, 
            dataset,  #    
            epochs,       # number of trials           
            accuracy_threshold = 1.0, 
            minibatch_size = 10,
            early_stop_after = 5):

        self.minibatch_size = minibatch_size
        self.train_accuracies = []
        self.test_accuracies = []
    
        best_validation_accuracy = 0.0
        best_test_accuracy = 0.0      
        best_loss = np.Infinity
        test_accuracy_with_best_loss = 0.0
        
        self.regression = dataset.regression
        
        num_train_minibatches = int(dataset.n_train / self.minibatch_size)
        self.num_train_minibatches = num_train_minibatches
        if dataset.validation_available == True:
            num_validation_minibatches = int(dataset.n_validation / self.minibatch_size)
        num_test_minibatches = int(dataset.n_test / self.minibatch_size)
        
        num_umimproved_epochs = 0
        sign = 1.0
        prev_loss = 0.0
        for i in range(epochs):            
            dataset.shuffle_train() 
            s = '{0}:'.format(i+1)
            print('Epoch',s.zfill(3), end=" ")
            start_time = time.time()
            self.loss = 0.0
            for j in range(num_train_minibatches):
                minibatch = dataset.get_batch('train', j, minibatch_size)
                self.__update_mini_batch__(minibatch)                
                self.__progress__(j+1, num_train_minibatches, 20, '=')                
            print(">", end=" ")           
            
            
            if self.regression == False:
                training_accuracy = np.mean([self.evaluate(dataset.get_batch('train', j, minibatch_size)) 
                                             for j in range(num_train_minibatches)])    
                self.train_accuracies.append(training_accuracy)            
                
            t = time.time()-start_time
            
            if dataset.test_available == True:
                if self.regression == False:
                    test_accuracy = np.mean([self.evaluate(dataset.get_batch('test', j, minibatch_size)) 
                                             for j in range(num_test_minibatches)])
            
                    self.test_accuracies.append(test_accuracy)                                
                    print("loss: {0:.5f} train_acc = {1:.2%} test_acc = {2:.2%} time: {3:.2f}s".format(self.loss, 
                                                                                                       training_accuracy,
                                                                                                       test_accuracy,                                                                                                                              t))               
                    if(test_accuracy >= best_test_accuracy):
                        best_test_accuracy = test_accuracy
                else:
                    print("loss: {0:.5f} time: {1:.2f}s".format(self.loss, t))               
            
            if self.loss < best_loss:
                # Optimal loss scenario
                best_loss = self.loss
                num_umimproved_epochs = 0
                if self.regression == False:
                    test_accuracy_with_best_loss = test_accuracy
            elif self.loss >= prev_loss:
                num_umimproved_epochs += 1 
            elif self.loss < prev_loss:
                num_umimproved_epochs = 0 
                
            prev_loss = self.loss
            
            if num_umimproved_epochs >= early_stop_after and i<epochs-1:
                print('Terminating early')
                break
        
            if self.regression == False:
                if training_accuracy >= accuracy_threshold:
                    print('Terminating early (training accuracy threshold reached)')
                    break 
            
            
        if self.regression == False:
            print("Accuracy: Maximum={0:.2%}; With optimal loss={1:.2%}".format(best_test_accuracy, 
                                                                                test_accuracy_with_best_loss))
    
    def evaluate(self, data):  
        X, y = data
        p = self.predict(X)
        tp = np.argmax(p, axis=-1)
        ty = np.argmax(y, axis=-1) 
        return np.mean(np.sum(ty == tp, axis=-1)/ty.shape[-1]) 

    def predict(self, x):        
        return (self.__feedforward__(np.array(x), False))
        
