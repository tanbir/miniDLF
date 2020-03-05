# miniDLF (Mini Deep Learning Framework)
From scratch using Numpy 

# Introduction

This is a humble attempt (inspired by [Stanford CS class CS231n](http://cs231n.github.io/)) to implement a simple Numpy-based <a hraf=https://keras.io/ target=blank>Keras</a>-like framework for designing Neural Networks. This is not intended to be used for your production environment. But feel free to use it for learning and development. Plenty of room for code-optimization.  

# Content
* [1 Network architectures](#1-network-architectures)
* [2 Layers](#2-layers)
  * [2.1 Core layers](#21-core-layers)
  * [2.2 Recurrent layers](#22-recurrent-layers)  
  * [2.3 Activation layers](#23-activations)    
* [3 Optimization algorithms](#3-optimization-algorithms)    
* [4 Model function: Sequential](#4-model-function-sequential)    
* [5 Some example runs](#5-some-example-runs)    
  * [5.1 Multilayer Perceptron using MNIST](#51-multilayer-perceptron-using-mnist)
  * [5.2 Convolutional Network using MNIST](#52-convolutional-network-using-mnist)
  * [5.3 Recurrent Networks using RNN, LSTM and GRU](#53-recurrent-neural-networks)

# 1 Network architectures
* Feedforward Neural Network 
* Convolutional Neural Network 
* Recurrent Neural Network
* Gated Recurrent Units (GRU)
* Long Short-Term Memory Units (LSTM)

# 2 Layers
## 2.1 Core layers
| Layer                    | Syntex to create an object                                                       |
|:-------------------------|:---------------------------------------------------------------------------------|
| Dense                    | `Dense(size, input_shape=None, trainable=True)`                                  |
| Conv2D                   | `Conv2D(filters, kernel, stride=1, padding=0, input_shape=None, trainable=True)` |
| Pooling2D                | `Pooling2D(size=2, stride=2, mode='max')`                                        |
| UpSampling2D             | `UpSampling2D(size=(y,x))`                                                       |
| Flatten                  | `Flatten()`                                                                      |
| Reshape                  | `Reshape(new_shape)`                                                             |
| Dropout                  | `Dropout(p_dropout)`                                                             |
| BatchNormalization       | `BatchNormalization(momentum=.9, scale=True, center=True)`                       |

## 2.2 Recurrent layers
| Layer                    | Syntex to create an object                                                       |
|:-------------------------|:---------------------------------------------------------------------------------|
| RNN                      | `RNN(n_units, input_shape = None)`                                  |
| LSTM                     | `LSTM(n_units, input_shape = None)`                                              |
| GRU                      | `GRU(n_units, input_shape = None)`                                               |

## 2.3 Activations
| Layer                    | Syntex to create an object                                                       |
|:-------------------------|:---------------------------------------------------------------------------------|
| Activation               | `Activation(activation, alpha=0.0001, max_value=2.5, scale=1.0)`                 |
| ReLU (activation)        | `ReLU(alpha=0.0, max_value=np.Infinity)`                                         |
| Leaky-ReLU (activation)  | `ReLU(alpha, max_value)`                                                         |
| ELU (activation)         | `ELU(alpha, max_value)`                                                          |
| Sigmoid (activation)     | `Sigmoid()`                                                                      |
| Tanh (activation)        | `Tanh()`                                                                         |
| Softmax (activation)     | `Softmax()`                                                                      |

# 3 Optimization algorithms
| Optimizer | Syntex to create an object                                              |
|:----------|:------------------------------------------------------------------------|
| SGD       | `SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)`                 |
| Adagrad   | `Adagrad(lr=0.01, decay=1e-6, epsilon=1e-7)`                            |
| RMSProp   | `RMSProp(lr=0.001, decay=1e-6, rho=0.9, epsilon=1e-6)`                  |
| Adam      | `Adam(lr=0.001, decay=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-6)`    |
| Nadam     | `Nadam(lr=0.001, decay=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-6)`   |
| AMSGrad   | `AMSGrad(lr=0.001, decay=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-6)` |

# 4 Model function: Sequential
| Function | Description                                                                       |
|:---------|:----------------------------------------------------------------------------------|
| Add      | `.add(layer)`                                                                     |
| Compile  | `.compile(loss = 'cce')`                                                          |
| Summary  | `.summary()`                                                                      |
| Train    | `.fit(dataset, epochs, minibatch_size = 10, early_stop_after = 5, optimizer='amsgrad')`|

* The parameter `layer` in the `add(...)` function is a layer-type object 
* The parameter `loss` in the `compile(...)` function is one of 'mse' / 'cce' / 'bce'
* The parameter `optimizer` in the `fit(...)` function can be either 
  * an object with specified or default parameters or 
  * a value from 'sgd'/'adagrad'/'rmsprop'/'adam'/'nadam'/'amsgrad'/'adadelta'

# 5 Some example runs
## 5.1 Multilayer Perceptron using MNIST
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
 
### CPU output 
    Dense      :   input_shape =  [784]  output_shape =  (400,)  trainable parameters =  314000
    Activation :   input_shape =  (400,)  output_shape =  (400,)  trainable parameters =  0
    Dropout    :   input_shape =  (400,)  output_shape =  (400,)  trainable parameters =  0
    Dense      :   input_shape =  (400,)  output_shape =  (400,)  trainable parameters =  160400
    Activation :   input_shape =  (400,)  output_shape =  (400,)  trainable parameters =  0
    Dropout    :   input_shape =  (400,)  output_shape =  (400,)  trainable parameters =  0
    Dense      :   input_shape =  (400,)  output_shape =  (400,)  trainable parameters =  160400
    Activation :   input_shape =  (400,)  output_shape =  (400,)  trainable parameters =  0
    Dropout    :   input_shape =  (400,)  output_shape =  (400,)  trainable parameters =  0
    Dense      :   input_shape =  (400,)  output_shape =  (10,)  trainable parameters =  4010
    Activation :   input_shape =  (10,)  output_shape =  (10,)  trainable parameters =  0
    Total # trainable parameters: 638810    

    Epoch 01: ====================> loss: 2.03288 train_acc = 87.32% test_acc = 87.98% time: 26.75s
    Epoch 02: ====================> loss: 0.36526 train_acc = 91.98% test_acc = 92.62% time: 25.25s
    Epoch 03: ====================> loss: 0.23474 train_acc = 94.33% test_acc = 94.21% time: 24.82s
    Epoch 04: ====================> loss: 0.18120 train_acc = 94.99% test_acc = 95.02% time: 24.28s
    Epoch 05: ====================> loss: 0.15364 train_acc = 95.96% test_acc = 95.63% time: 23.91s
    Epoch 06: ====================> loss: 0.12797 train_acc = 96.29% test_acc = 95.68% time: 24.35s
    Epoch 07: ====================> loss: 0.11399 train_acc = 96.94% test_acc = 96.39% time: 23.33s
    Epoch 08: ====================> loss: 0.09870 train_acc = 97.26% test_acc = 96.37% time: 23.97s
    Epoch 09: ====================> loss: 0.09071 train_acc = 97.59% test_acc = 96.72% time: 26.06s
    Epoch 10: ====================> loss: 0.07748 train_acc = 97.75% test_acc = 96.95% time: 23.73s
    Epoch 11: ====================> loss: 0.07103 train_acc = 97.93% test_acc = 96.80% time: 24.08s
    Epoch 12: ====================> loss: 0.06592 train_acc = 98.11% test_acc = 96.98% time: 24.05s
    Epoch 13: ====================> loss: 0.05985 train_acc = 98.35% test_acc = 96.75% time: 26.55s
    Epoch 14: ====================> loss: 0.05286 train_acc = 98.33% test_acc = 97.13% time: 26.24s
    Epoch 15: ====================> loss: 0.04977 train_acc = 98.57% test_acc = 97.35% time: 23.85s
    Epoch 16: ====================> loss: 0.04704 train_acc = 98.54% test_acc = 97.26% time: 24.55s
    Epoch 17: ====================> loss: 0.04373 train_acc = 98.69% test_acc = 97.20% time: 23.78s
    Epoch 18: ====================> loss: 0.04225 train_acc = 98.77% test_acc = 97.29% time: 23.55s
    Epoch 19: ====================> loss: 0.03697 train_acc = 98.97% test_acc = 97.64% time: 22.62s
    Epoch 20: ====================> loss: 0.03574 train_acc = 98.92% test_acc = 97.32% time: 24.08s
    Epoch 21: ====================> loss: 0.03627 train_acc = 99.01% test_acc = 97.50% time: 23.51s
    Epoch 22: ====================> loss: 0.03262 train_acc = 99.00% test_acc = 97.39% time: 23.57s
    Epoch 23: ====================> loss: 0.03035 train_acc = 99.10% test_acc = 97.55% time: 23.21s
    Epoch 24: ====================> loss: 0.03081 train_acc = 99.05% test_acc = 97.36% time: 23.40s
    Epoch 25: ====================> loss: 0.02681 train_acc = 99.09% test_acc = 97.44% time: 23.03s
    Accuracy: Maximum=97.64%; With optimal loss=97.44%

## 5.2 Convolutional Network using MNIST
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

    m.fit(dataset=mnist, epochs=15, minibatch_size = 512, early_stop_after = 10)
 
### CPU output 
    Conv2D     :   input_shape =  [ 1 28 28]  output_shape =  (32, 12, 12)  trainable parameters =  1184
    Activation :   input_shape =  (32, 12, 12)  output_shape =  (32, 12, 12)  trainable parameters =  0
    Pooling    :   input_shape =  (32, 12, 12)  output_shape =  (32, 6, 6)  trainable parameters =  0
    Conv2D     :   input_shape =  (32, 6, 6)  output_shape =  (32, 2, 2)  trainable parameters =  9248
    Activation :   input_shape =  (32, 2, 2)  output_shape =  (32, 2, 2)  trainable parameters =  0
    Pooling    :   input_shape =  (32, 2, 2)  output_shape =  (32, 1, 1)  trainable parameters =  0
    Flatten    :   input_shape =  (32, 1, 1)  output_shape =  (32,)  trainable parameters =  0
    Dense      :   input_shape =  (32,)  output_shape =  (100,)  trainable parameters =  3300
    Activation :   input_shape =  (100,)  output_shape =  (100,)  trainable parameters =  0
    BatchNorm  :   input_shape =  (100,)  output_shape =  (100,)  trainable parameters =  200
    Dense      :   input_shape =  (100,)  output_shape =  (10,)  trainable parameters =  1010
    Activation :   input_shape =  (10,)  output_shape =  (10,)  trainable parameters =  0
    Total # trainable parameters: 14942

    Epoch 01: ====================> loss: 1.08035 train_acc = 85.71% test_acc = 87.06% time: 185.34s
    Epoch 02: ====================> loss: 0.33217 train_acc = 92.64% test_acc = 93.25% time: 185.48s
    Epoch 03: ====================> loss: 0.21521 train_acc = 94.56% test_acc = 95.08% time: 188.01s
    Epoch 04: ====================> loss: 0.16735 train_acc = 95.65% test_acc = 95.80% time: 187.31s
    Epoch 05: ====================> loss: 0.14283 train_acc = 96.14% test_acc = 96.38% time: 189.34s
    Epoch 06: ====================> loss: 0.12652 train_acc = 96.53% test_acc = 96.64% time: 187.18s
    Epoch 07: ====================> loss: 0.11313 train_acc = 96.77% test_acc = 96.78% time: 189.46s
    Epoch 08: ====================> loss: 0.17498 train_acc = 92.46% test_acc = 93.00% time: 188.45s
    Epoch 09: ====================> loss: 0.15132 train_acc = 96.22% test_acc = 96.12% time: 186.56s
    Epoch 10: ====================> loss: 0.11911 train_acc = 96.66% test_acc = 96.86% time: 187.46s
    Epoch 11: ====================> loss: 0.10841 train_acc = 96.88% test_acc = 96.79% time: 186.77s
    Epoch 12: ====================> loss: 0.10173 train_acc = 97.12% test_acc = 97.00% time: 190.63s
    Epoch 13: ====================> loss: 0.09665 train_acc = 97.22% test_acc = 97.26% time: 190.30s
    Epoch 14: ====================> loss: 0.09284 train_acc = 97.39% test_acc = 97.16% time: 192.27s
    Epoch 15: ====================> loss: 0.11496 train_acc = 97.24% test_acc = 97.36% time: 187.97s
    Accuracy: Maximum=97.36%; With optimal loss=97.16%

## 5.3 Recurrent Neural Networks

### RNN
    import numpy as np
    from miniDLF.models import Sequential
    from miniDLF.layers import RNN, Activation
    from miniDLF.optimizers import Adam
    from miniDLF.datasets import TEXT2SEQ

    d = TEXT2SEQ('./data/TEXT/basic_rnn_from_wiki.txt', 60)

    m = Sequential() 
    m.add(RNN(256, input_shape=d.input_shape))
    m.add(RNN(256, input_shape=d.input_shape))
    m.add(RNN(256, input_shape=d.input_shape))

    m.add(Activation('softmax'))
    m.compile(loss='cce', optimizer=Adam())
    m.summary()

    m.fit(dataset=d, epochs=1000, minibatch_size = 25, accuracy_threshold=0.95, early_stop_after = 30)
    
#### CPU output (stacked RNNs)
    RNN        :   input_shape =  [60 43]  output_shape =  [60 43]  trainable parameters =  87851
    RNN        :   input_shape =  [60 43]  output_shape =  [60 43]  trainable parameters =  87851
    RNN        :   input_shape =  [60 43]  output_shape =  [60 43]  trainable parameters =  87851
    Activation :   input_shape =  [60 43]  output_shape =  [60 43]  trainable parameters =  0
    Total # trainable parameters: 263553

    Epoch 01: ====================> loss: 59.83458 train_acc = 53.63% test_acc = 52.94% time: 18.58s
    Epoch 02: ====================> loss: 29.94776 train_acc = 78.03% test_acc = 77.09% time: 18.40s
    Epoch 03: ====================> loss: 16.57419 train_acc = 89.55% test_acc = 88.94% time: 18.29s
    Epoch 04: ====================> loss: 10.08636 train_acc = 93.15% test_acc = 92.35% time: 18.21s
    Epoch 05: ====================> loss: 7.37533 train_acc = 94.12% test_acc = 93.23% time: 18.14s
    Epoch 06: ====================> loss: 6.11092 train_acc = 94.71% test_acc = 93.62% time: 17.97s
    Epoch 07: ====================> loss: 5.39310 train_acc = 95.22% test_acc = 93.95% time: 17.87s
    Terminating early (training accuracy threshold reached)
    Accuracy: Maximum=93.95%; With optimal loss=93.95%
    
#### Some sequence outputs
    X = ences its input stream through output units connected to act
    y = nces its input stream through output units connected to actu
    p =   e  ons input stream through output units connected to actu

    X = with a directed (one-way) connection to every other node in 
    y = ith a directed (one-way) connection to every other node in t
    p = evh t directed (one-way) connection to every other node in t

    X =  progress is measured with the number of points won. Each se
    y = progress is measured with the number of points won. Each seq
    p = aeomhess is measured with the number of points won. Each seq


### LSTM
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

#### CPU output (LSTM)
    LSTM       :   input_shape =  [60 43]  output_shape =  [60 43]  trainable parameters =  318251
    Activation :   input_shape =  [60 43]  output_shape =  [60 43]  trainable parameters =  0
    Total # trainable parameters: 318251

    Epoch 01: ====================> loss: 72.30946 train_acc = 30.97% test_acc = 30.48% time: 24.50s
    Epoch 02: ====================> loss: 56.34389 train_acc = 40.57% test_acc = 40.16% time: 24.51s
    Epoch 03: ====================> loss: 47.43393 train_acc = 48.78% test_acc = 48.11% time: 24.35s
    Epoch 04: ====================> loss: 41.47037 train_acc = 54.77% test_acc = 54.50% time: 24.75s
    Epoch 05: ====================> loss: 36.94661 train_acc = 59.81% test_acc = 58.54% time: 24.42s
    Epoch 06: ====================> loss: 31.98623 train_acc = 64.58% test_acc = 63.66% time: 24.27s
    Epoch 07: ====================> loss: 27.54223 train_acc = 71.11% test_acc = 69.98% time: 24.05s
    Epoch 08: ====================> loss: 24.12392 train_acc = 75.93% test_acc = 74.64% time: 23.96s
    Epoch 09: ====================> loss: 20.62332 train_acc = 78.69% test_acc = 77.67% time: 24.61s
    Epoch 10: ====================> loss: 18.22262 train_acc = 81.49% test_acc = 80.41% time: 24.65s
    Epoch 11: ====================> loss: 15.25191 train_acc = 86.28% test_acc = 85.27% time: 25.27s
    Epoch 12: ====================> loss: 13.28545 train_acc = 88.03% test_acc = 86.88% time: 24.88s
    Epoch 13: ====================> loss: 11.83956 train_acc = 90.73% test_acc = 89.88% time: 24.97s
    Epoch 14: ====================> loss: 9.74214 train_acc = 91.60% test_acc = 90.91% time: 24.95s
    Epoch 15: ====================> loss: 9.19856 train_acc = 91.82% test_acc = 91.29% time: 24.66s
    Epoch 16: ====================> loss: 7.86698 train_acc = 93.56% test_acc = 92.98% time: 24.50s
    Epoch 17: ====================> loss: 7.16971 train_acc = 93.36% test_acc = 92.62% time: 24.88s
    Epoch 18: ====================> loss: 6.82040 train_acc = 93.87% test_acc = 93.05% time: 24.65s
    Epoch 19: ====================> loss: 6.26345 train_acc = 94.31% test_acc = 93.59% time: 24.77s
    Epoch 20: ====================> loss: 5.83023 train_acc = 94.51% test_acc = 93.83% time: 24.82s
    Epoch 21: ====================> loss: 5.45822 train_acc = 94.74% test_acc = 93.99% time: 24.90s
    Epoch 22: ====================> loss: 5.33980 train_acc = 94.98% test_acc = 94.22% time: 24.56s
    Epoch 23: ====================> loss: 4.89512 train_acc = 95.27% test_acc = 94.41% time: 24.61s
    Terminating early (training accuracy threshold reached)
    Accuracy: Maximum=94.41%; With optimal loss=94.41%

### GRU
    import numpy as np
    from miniDLF.models import Sequential
    from miniDLF.layers import GRU, Activation
    from miniDLF.optimizers import Adam
    from miniDLF.datasets import TEXT2SEQ

    d = TEXT2SEQ('./data/TEXT/basic_rnn_from_wiki.txt', 60)

    m = Sequential() 
    m.add(GRU(256, input_shape=d.input_shape))    

    m.add(Activation('softmax'))
    m.compile(loss='cce', optimizer=Adam())
    m.summary()
    
    m.fit(dataset=d, epochs=1000, minibatch_size = 25, accuracy_threshold=0.95, early_stop_after = 30)
    
#### CPU output (GRUs)    
    GRU        :   input_shape =  [60 43]  output_shape =  [60 43]  trainable parameters =  241451
    Activation :   input_shape =  [60 43]  output_shape =  [60 43]  trainable parameters =  0
    Total # trainable parameters: 241451
    
    Epoch 01: ====================> loss: 61.40361 train_acc = 49.84% test_acc = 49.91% time: 18.78s
    Epoch 02: ====================> loss: 36.44772 train_acc = 68.11% test_acc = 68.13% time: 18.70s
    Epoch 03: ====================> loss: 23.07642 train_acc = 81.48% test_acc = 81.07% time: 19.51s
    Epoch 04: ====================> loss: 14.29644 train_acc = 90.41% test_acc = 89.94% time: 21.21s
    Epoch 05: ====================> loss: 9.26848 train_acc = 93.22% test_acc = 92.83% time: 19.77s
    Epoch 06: ====================> loss: 6.98398 train_acc = 94.34% test_acc = 94.08% time: 19.18s
    Epoch 07: ====================> loss: 5.96353 train_acc = 94.86% test_acc = 94.32% time: 21.44s
    Epoch 08: ====================> loss: 5.32681 train_acc = 95.13% test_acc = 94.70% time: 20.39s
    Terminating early (training accuracy threshold reached)
    Accuracy: Maximum=94.70%; With optimal loss=94.70%
              
