import numpy as np
from miniDLF.models import Sequential
from miniDLF.layers import LSTM, Activation
from miniDLF.optimizers import Adam
from miniDLF.datasets import TEXT2SEQ

d = TEXT2SEQ('./data/TEXT/basic_rnn_from_wiki.txt', 60)

m = Sequential() 
m.add(LSTM(256, input_shape=d.input_shape))

m.add(Activation('softmax'))
m.compile(loss='cce', optimizer=Adam())
m.summary()

m.fit(dataset=d, epochs=1000, minibatch_size = 25, accuracy_threshold=0.95, early_stop_after = 30)