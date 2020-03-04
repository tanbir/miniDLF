import numpy as np
from miniDLF.models import Sequential
from miniDLF.layers import LSTM, Activation
from miniDLF.optimizers import Adam
from miniDLF.datasets import TEXT2SEQ



d = TEXT2SEQ('./data/TEXT/basic_rnn.txt', 50)

m = Sequential() 
m.add(LSTM(512, input_shape=d.input_shape))
m.add(Activation('softmax'))
m.compile(loss='cce', optimizer=Adam())
m.summary()



m.fit(dataset=d, epochs=1000, minibatch_size = 50, accuracy_threshold=0.96, early_stop_after = 30)