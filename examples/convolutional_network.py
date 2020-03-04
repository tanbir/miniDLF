from miniDLF.models import Sequential
from miniDLF.layers import *
from miniDLF.optimizers import *
from miniDLF.losses import *
from miniDLF.datasets import *


mnist = MNIST('./data/MNIST/mnist.pkl.gz', n_classes=10, input_shape=(1, 28, 28))

m = Sequential() 
m.add(Conv2D(32, (6,6), 2, input_shape=(1, 28, 28)))
m.add(Activation('leaky_relu'))
m.add(Pooling2D(2, 2, 'avg'))
m.add(Conv2D(32, (3,3), 2))
m.add(Activation('leaky_relu'))
m.add(Pooling2D(2, 2, 'avg'))
m.add(Flatten())
m.add(Dense(100))
m.add(Activation('leaky_relu'))
m.add(BatchNormalization())
m.add(Dense(10))
m.add(Activation('softmax'))
m.compile(loss='cce', optimizer=Adam())
m.summary()

m.fit(dataset=mnist, epochs=13, minibatch_size = 512, early_stop_after = 10)