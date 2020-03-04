from miniDLF.models import Sequential
from miniDLF.layers import Dense, Dropout, Activation
from miniDLF.optimizers import Adam
from miniDLF.datasets import MNIST

mnist = MNIST('./data/MNIST/mnist.pkl.gz', n_classes=10, input_shape=(784,))

m = Sequential()
m.add(Dense(400, input_shape=(784,)))
m.add(Activation('leaky_relu'))
m.add(Dropout(0.2))
m.add(Dense(400))
m.add(Activation('leaky_relu'))
m.add(Dropout(0.2))
m.add(Dense(400))
m.add(Activation('leaky_relu'))
m.add(Dropout(0.2))
m.add(Dense(10))
m.add(Activation('softmax'))
m.compile(loss='cce', optimizer=Adam())
m.summary()

m.fit(dataset=mnist, epochs=25, minibatch_size = 512, early_stop_after = 10)