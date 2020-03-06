from miniDLF.models import Sequential, Autoencoder

from miniDLF.layers import Dense, Activation
from miniDLF.datasets import Dataset
import gzip, pickle
import numpy as np

enc = Sequential()
enc.add(Dense(100, input_shape=(784,)))
enc.add(Activation('leaky_relu'))
enc.add(Dense(64))


dec = Sequential()
dec.add(Dense(64, input_shape=(64,)))
dec.add(Activation('leaky_relu'))
dec.add(Dense(784))
dec.add(Activation('sigmoid'))

ae = Autoencoder(enc, dec, loss='bce', optimizer='adam')
ae.summary()


f = gzip.open('./data/MNIST/mnist.pkl.gz', 'rb')
train, validation, test = pickle.load(f, encoding='latin1')  
t0 = np.array(train[0])
v0 = np.array(validation[0])            
c0 = np.concatenate((t0, v0), axis=0)    
train = c0, c0
                
test = np.array(test[0]), np.array(test[0])    
               
d = Dataset(train, None, test, (784,))

ae.fit(dataset=d, epochs=1000, minibatch_size = 256, accuracy_threshold=0.96, early_stop_after = 10)
