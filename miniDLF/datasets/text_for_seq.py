from .base import Dataset
import numpy as np

class TEXT2SEQ(Dataset):
    def __init__(self, path, timesteps):
        self.path = path  
        self.input_shape = None
        self.n_classes = None
        
        f = open(self.path, 'r')
        self.txt = f.read()        
        X, y = self.get_Xy(self.txt, timesteps)
        X_train, X_test, y_train, y_test = self.train_test_split(X, y, test_size=0.2)
        Dataset.__init__(self, [X_train, y_train], None, [np.array(X_test), np.array(y_test)], 0, input_shape=(X.shape[1], X.shape[2]))
        
        self.input_shape = (X.shape[1], X.shape[2])
        self.n_classes = X.shape[2]
        f.close()                        
            
    def get_Xy(self, txt, timesteps):
        n = len(txt)
        char_to_idx = {char: i for i, char in enumerate(set(txt))}
        idx_to_char = {i: char for i, char in enumerate(set(txt))}
        input_dim = len(char_to_idx)
        
        
        X = np.zeros([n-timesteps, timesteps, input_dim], dtype=float)
        y = np.zeros([n-timesteps, timesteps, input_dim], dtype=float)
        for i in range(n-timesteps):
            a = txt[i:i+timesteps]
            b = txt[i+1:i+timesteps+1]
            a1 = [char_to_idx[a[j]] for j in range(timesteps)]
            b1 = [char_to_idx[b[j]] for j in range(timesteps)]
            X[i] = np.array([self.__onehot__(a1[j], input_dim) for j in range(timesteps)]) 
            y[i] = np.array([self.__onehot__(b1[j], input_dim) for j in range(timesteps)])     
        return X, y        
        
        
       
