# miniDLF 
(Mini Deep Learning Framework)

# Introduction

This is a humble attempt (inspired by [Stanford CS class CS231n](http://cs231n.github.io/)) to implement a simple Numpy-based <a hraf=https://keras.io/ target=blank>Keras</a>-like framework for designing Neural Networks. This is not intended to be used for your production environment. But feel free to use it for learning and development. Plenty of room for code-optimization.  

# Content
* [1 Network architectures](#1-network-architectures)
* [2 Layers](#2-layers)
  * [2.1 Core layers](#21-core-layers)
  * [2.2 Recurrent layers](#22-recurrent-layers)  
  * [2.3 Activation layers](#23-activations)    
* [3 Optimization algorithms](#3-optimization-algorithms)    

# 1 Network architectures
* Feedforward Neural Network 
* Convolutional Neural Network 
* Recurrent Neural Network
* Gated Recurrent Units (GRU)
* Long Short Term Memomry Units (LSTM)

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

    Epoch 01: ===================================> loss: 2.03288 train_acc = 87.32% test_acc = 87.98% time: 26.75s
    Epoch 02: ===================================> loss: 0.36526 train_acc = 91.98% test_acc = 92.62% time: 25.25s
    Epoch 03: ===================================> loss: 0.23474 train_acc = 94.33% test_acc = 94.21% time: 24.82s
    Epoch 04: ===================================> loss: 0.18120 train_acc = 94.99% test_acc = 95.02% time: 24.28s
    Epoch 05: ===================================> loss: 0.15364 train_acc = 95.96% test_acc = 95.63% time: 23.91s
    Epoch 06: ===================================> loss: 0.12797 train_acc = 96.29% test_acc = 95.68% time: 24.35s
    Epoch 07: ===================================> loss: 0.11399 train_acc = 96.94% test_acc = 96.39% time: 23.33s
    Epoch 08: ===================================> loss: 0.09870 train_acc = 97.26% test_acc = 96.37% time: 23.97s
    Epoch 09: ===================================> loss: 0.09071 train_acc = 97.59% test_acc = 96.72% time: 26.06s
    Epoch 10: ===================================> loss: 0.07748 train_acc = 97.75% test_acc = 96.95% time: 23.73s
    Epoch 11: ===================================> loss: 0.07103 train_acc = 97.93% test_acc = 96.80% time: 24.08s
    Epoch 12: ===================================> loss: 0.06592 train_acc = 98.11% test_acc = 96.98% time: 24.05s
    Epoch 13: ===================================> loss: 0.05985 train_acc = 98.35% test_acc = 96.75% time: 26.55s
    Epoch 14: ===================================> loss: 0.05286 train_acc = 98.33% test_acc = 97.13% time: 26.24s
    Epoch 15: ===================================> loss: 0.04977 train_acc = 98.57% test_acc = 97.35% time: 23.85s
    Epoch 16: ===================================> loss: 0.04704 train_acc = 98.54% test_acc = 97.26% time: 24.55s
    Epoch 17: ===================================> loss: 0.04373 train_acc = 98.69% test_acc = 97.20% time: 23.78s
    Epoch 18: ===================================> loss: 0.04225 train_acc = 98.77% test_acc = 97.29% time: 23.55s
    Epoch 19: ===================================> loss: 0.03697 train_acc = 98.97% test_acc = 97.64% time: 22.62s
    Epoch 20: ===================================> loss: 0.03574 train_acc = 98.92% test_acc = 97.32% time: 24.08s
    Epoch 21: ===================================> loss: 0.03627 train_acc = 99.01% test_acc = 97.50% time: 23.51s
    Epoch 22: ===================================> loss: 0.03262 train_acc = 99.00% test_acc = 97.39% time: 23.57s
    Epoch 23: ===================================> loss: 0.03035 train_acc = 99.10% test_acc = 97.55% time: 23.21s
    Epoch 24: ===================================> loss: 0.03081 train_acc = 99.05% test_acc = 97.36% time: 23.40s
    Epoch 25: ===================================> loss: 0.02681 train_acc = 99.09% test_acc = 97.44% time: 23.03s
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

    Epoch 01: ==================================> loss: 1.08035 train_acc = 85.71% test_acc = 87.06% time: 185.34s
    Epoch 02: ==================================> loss: 0.33217 train_acc = 92.64% test_acc = 93.25% time: 185.48s
    Epoch 03: ==================================> loss: 0.21521 train_acc = 94.56% test_acc = 95.08% time: 188.01s
    Epoch 04: ==================================> loss: 0.16735 train_acc = 95.65% test_acc = 95.80% time: 187.31s
    Epoch 05: ==================================> loss: 0.14283 train_acc = 96.14% test_acc = 96.38% time: 189.34s
    Epoch 06: ==================================> loss: 0.12652 train_acc = 96.53% test_acc = 96.64% time: 187.18s
    Epoch 07: ==================================> loss: 0.11313 train_acc = 96.77% test_acc = 96.78% time: 189.46s
    Epoch 08: ==================================> loss: 0.17498 train_acc = 92.46% test_acc = 93.00% time: 188.45s
    Epoch 09: ==================================> loss: 0.15132 train_acc = 96.22% test_acc = 96.12% time: 186.56s
    Epoch 10: ==================================> loss: 0.11911 train_acc = 96.66% test_acc = 96.86% time: 187.46s
    Epoch 11: ==================================> loss: 0.10841 train_acc = 96.88% test_acc = 96.79% time: 186.77s
    Epoch 12: ==================================> loss: 0.10173 train_acc = 97.12% test_acc = 97.00% time: 190.63s
    Epoch 13: ==================================> loss: 0.09665 train_acc = 97.22% test_acc = 97.26% time: 190.30s
    Epoch 14: ==================================> loss: 0.09284 train_acc = 97.39% test_acc = 97.16% time: 192.27s
    Epoch 15: ==================================> loss: 0.11496 train_acc = 97.24% test_acc = 97.36% time: 187.97s
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
    m.add(RNN(256, input_shape=d.input_shape))
    m.add(RNN(256, input_shape=d.input_shape))

    m.add(Activation('softmax'))
    m.compile(loss='cce', optimizer=Adam())
    m.summary()

    m.fit(dataset=d, epochs=1000, minibatch_size = 25, accuracy_threshold=0.96, early_stop_after = 30)
    
#### CPU output (stacked RNNs)
    RNN        :   input_shape =  [60 41]  output_shape =  [60 41]  trainable parameters =  86825
    RNN        :   input_shape =  [60 41]  output_shape =  [60 41]  trainable parameters =  86825
    RNN        :   input_shape =  [60 41]  output_shape =  [60 41]  trainable parameters =  86825
    RNN        :   input_shape =  [60 41]  output_shape =  [60 41]  trainable parameters =  86825
    RNN        :   input_shape =  [60 41]  output_shape =  [60 41]  trainable parameters =  86825
    Activation :   input_shape =  [60 41]  output_shape =  [60 41]  trainable parameters =  0
    Total # trainable parameters: 434125

    Epoch 01: ==========================> loss: 67.07430 train_acc = 43.86% test_acc = 43.69% time: 18.24s
    Epoch 02: ==========================> loss: 33.96766 train_acc = 76.54% test_acc = 76.46% time: 18.17s
    Epoch 03: ==========================> loss: 14.29152 train_acc = 91.95% test_acc = 91.74% time: 18.20s
    Epoch 04: ==========================> loss: 7.30623 train_acc = 94.94% test_acc = 94.20% time: 17.82s
    Epoch 05: ==========================> loss: 5.45037 train_acc = 95.42% test_acc = 94.70% time: 17.81s
    Epoch 06: ==========================> loss: 4.60923 train_acc = 95.75% test_acc = 94.84% time: 18.28s
    Epoch 07: ==========================> loss: 4.04447 train_acc = 96.03% test_acc = 94.98% time: 18.19s
    Terminating early (training accuracy threshold reached)
    Accuracy: Maximum=94.98%; With optimal loss=94.98%
    
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

    d = TEXT2SEQ('./data/TEXT/basic_rnn.txt', 50)

    m = Sequential() 
    m.add(LSTM(512, input_shape=d.input_shape))
    m.add(Activation('softmax'))
    m.compile(loss='cce', optimizer=Adam())
    m.summary()

    m.fit(dataset=d, epochs=1000, minibatch_size = 50, accuracy_threshold=0.96, early_stop_after = 30)

#### CPU output (LSTM)
    LSTM       :   input_shape =  [50 41]  output_shape =  [50 41]  trainable parameters =  1155625
    Activation :   input_shape =  [50 41]  output_shape =  [50 41]  trainable parameters =  0
    Total # trainable parameters: 1155625

    Epoch 01: ==========================> loss: 153.75808 train_acc = 21.28% test_acc = 20.88% time: 29.65s
    Epoch 02: ==========================> loss: 129.60586 train_acc = 33.58% test_acc = 33.84% time: 29.57s
    Epoch 03: ==========================> loss: 113.64453 train_acc = 38.87% test_acc = 38.73% time: 29.66s
    Epoch 04: ==========================> loss: 100.89265 train_acc = 43.66% test_acc = 43.02% time: 29.25s
    Epoch 05: ==========================> loss: 92.87335 train_acc = 48.32% test_acc = 47.88% time: 28.97s
    Epoch 06: ==========================> loss: 84.49295 train_acc = 53.43% test_acc = 53.05% time: 29.23s
    Epoch 07: ==========================> loss: 76.75869 train_acc = 57.31% test_acc = 56.40% time: 30.26s
    Epoch 08: ==========================> loss: 68.96725 train_acc = 60.00% test_acc = 59.16% time: 32.14s
    Epoch 09: ==========================> loss: 67.36865 train_acc = 62.40% test_acc = 61.69% time: 31.19s
    Epoch 10: ==========================> loss: 60.31191 train_acc = 68.17% test_acc = 67.25% time: 30.98s
    Epoch 11: ==========================> loss: 53.84947 train_acc = 69.45% test_acc = 68.81% time: 31.54s
    Epoch 12: ==========================> loss: 49.32982 train_acc = 74.92% test_acc = 74.09% time: 30.72s
    Epoch 13: ==========================> loss: 44.37329 train_acc = 74.06% test_acc = 73.21% time: 31.58s
    Epoch 14: ==========================> loss: 41.45781 train_acc = 80.24% test_acc = 79.23% time: 31.76s
    Epoch 15: ==========================> loss: 35.81229 train_acc = 83.14% test_acc = 81.99% time: 30.83s
    Epoch 16: ==========================> loss: 32.65397 train_acc = 86.32% test_acc = 85.59% time: 31.28s
    Epoch 17: ==========================> loss: 27.72565 train_acc = 87.78% test_acc = 87.24% time: 31.58s
    Epoch 18: ==========================> loss: 25.07848 train_acc = 89.64% test_acc = 88.61% time: 30.95s
    Epoch 19: ==========================> loss: 21.96866 train_acc = 90.55% test_acc = 89.74% time: 30.64s
    Epoch 20: ==========================> loss: 19.76275 train_acc = 91.08% test_acc = 90.32% time: 31.06s
    Epoch 21: ==========================> loss: 19.03882 train_acc = 91.50% test_acc = 90.56% time: 31.03s
    Epoch 22: ==========================> loss: 17.55323 train_acc = 91.83% test_acc = 91.45% time: 31.11s
    Epoch 23: ==========================> loss: 15.68028 train_acc = 93.39% test_acc = 92.68% time: 30.74s
    Epoch 24: ==========================> loss: 13.08973 train_acc = 93.91% test_acc = 93.22% time: 31.97s
    Epoch 25: ==========================> loss: 12.29995 train_acc = 94.27% test_acc = 93.41% time: 31.09s
    Epoch 26: ==========================> loss: 11.49621 train_acc = 94.66% test_acc = 93.76% time: 32.07s
    Epoch 27: ==========================> loss: 10.69406 train_acc = 94.88% test_acc = 93.99% time: 31.84s
    Epoch 28: ==========================> loss: 10.10716 train_acc = 94.92% test_acc = 93.89% time: 30.66s
    Epoch 29: ==========================> loss: 9.96864 train_acc = 95.05% test_acc = 94.09% time: 30.96s
    Epoch 30: ==========================> loss: 9.57908 train_acc = 95.16% test_acc = 94.20% time: 30.85s
    Epoch 31: ==========================> loss: 9.41546 train_acc = 95.21% test_acc = 94.19% time: 31.41s
    Epoch 32: ==========================> loss: 8.97033 train_acc = 95.32% test_acc = 94.26% time: 31.24s
    Epoch 33: ==========================> loss: 8.80891 train_acc = 95.44% test_acc = 94.25% time: 31.47s
    Epoch 34: ==========================> loss: 8.55762 train_acc = 95.45% test_acc = 94.34% time: 30.98s
    Epoch 35: ==========================> loss: 8.54328 train_acc = 95.52% test_acc = 94.41% time: 31.62s
    Epoch 36: ==========================> loss: 8.36699 train_acc = 95.52% test_acc = 94.39% time: 31.51s
    Epoch 37: ==========================> loss: 8.14359 train_acc = 95.62% test_acc = 94.59% time: 32.49s
    Epoch 38: ==========================> loss: 8.03375 train_acc = 95.65% test_acc = 94.57% time: 30.85s
    Epoch 39: ==========================> loss: 7.95894 train_acc = 95.62% test_acc = 94.47% time: 31.57s
    Epoch 40: ==========================> loss: 7.79742 train_acc = 95.72% test_acc = 94.53% time: 31.73s
    Epoch 41: ==========================> loss: 7.90038 train_acc = 95.76% test_acc = 94.41% time: 31.24s
    Epoch 42: ==========================> loss: 7.63093 train_acc = 95.71% test_acc = 94.49% time: 31.18s
    Epoch 43: ==========================> loss: 7.62404 train_acc = 95.84% test_acc = 94.49% time: 30.74s
    Epoch 44: ==========================> loss: 7.42451 train_acc = 95.88% test_acc = 94.74% time: 31.07s
    Epoch 45: ==========================> loss: 7.23362 train_acc = 95.89% test_acc = 94.67% time: 31.42s
    Epoch 46: ==========================> loss: 7.13295 train_acc = 95.90% test_acc = 94.50% time: 31.51s
    Epoch 47: ==========================> loss: 7.15260 train_acc = 95.90% test_acc = 94.54% time: 31.79s
    Epoch 48: ==========================> loss: 6.90957 train_acc = 95.99% test_acc = 94.50% time: 31.07s
    Epoch 49: ==========================> loss: 6.94683 train_acc = 96.03% test_acc = 94.61% time: 30.56s
    Terminating early (training accuracy threshold reached)
    Accuracy: Maximum=94.74%; With optimal loss=94.50%

### GRU
    import numpy as np
    from miniDLF.models import Sequential
    from miniDLF.layers import RNN, GRU, LSTM, Activation
    from miniDLF.optimizers import Adam
    from miniDLF.datasets import TEXT2SEQ

    d = TEXT2SEQ('./data/TEXT/basic_rnn.txt', 50)

    m = Sequential() 
    m.add(GRU(512, input_shape=d.input_shape))
    m.add(Activation('softmax'))
    m.compile(loss='cce', optimizer=Adam())
    m.summary()
    
    m.fit(dataset=d, epochs=1000, minibatch_size = 50, accuracy_threshold=0.96, early_stop_after = 30)
    
#### CPU output (GRU)    
    GRU        :   input_shape =  [50 41]  output_shape =  [50 41]  trainable parameters =  871977
    Activation :   input_shape =  [50 41]  output_shape =  [50 41]  trainable parameters =  0
    Total # trainable parameters: 871977
    
    Epoch 01: ==========================> loss: 129.07562 train_acc = 46.60% test_acc = 45.24% time: 22.44s
    Epoch 02: ==========================> loss: 80.18906 train_acc = 65.20% test_acc = 63.57% time: 22.29s
    Epoch 03: ==========================> loss: 51.41698 train_acc = 80.12% test_acc = 78.98% time: 21.96s
    Epoch 04: ==========================> loss: 31.13936 train_acc = 90.30% test_acc = 89.98% time: 23.05s
    Epoch 05: ==========================> loss: 18.97702 train_acc = 93.50% test_acc = 93.14% time: 22.12s
    Epoch 06: ==========================> loss: 13.28802 train_acc = 94.59% test_acc = 94.17% time: 22.40s
    Epoch 07: ==========================> loss: 10.79545 train_acc = 94.89% test_acc = 94.29% time: 22.07s
    Epoch 08: ==========================> loss: 9.43124 train_acc = 95.25% test_acc = 94.79% time: 22.42s
    Epoch 09: ==========================> loss: 8.62526 train_acc = 95.50% test_acc = 94.97% time: 22.21s
    Epoch 10: ==========================> loss: 8.04648 train_acc = 95.48% test_acc = 95.05% time: 22.24s
    Epoch 11: ==========================> loss: 7.63974 train_acc = 95.73% test_acc = 95.15% time: 23.60s
    Epoch 12: ==========================> loss: 7.30892 train_acc = 95.79% test_acc = 95.32% time: 21.91s
    Epoch 13: ==========================> loss: 7.06969 train_acc = 95.84% test_acc = 95.32% time: 22.05s
    Epoch 14: ==========================> loss: 6.83909 train_acc = 95.91% test_acc = 95.31% time: 21.87s
    Epoch 15: ==========================> loss: 6.68061 train_acc = 95.94% test_acc = 95.29% time: 21.81s
    Epoch 16: ==========================> loss: 6.49831 train_acc = 95.98% test_acc = 95.29% time: 23.24s
    Epoch 17: ==========================> loss: 6.39038 train_acc = 96.03% test_acc = 95.28% time: 22.56s
    Terminating early (training accuracy threshold reached)
    Accuracy: Maximum=95.32%; With optimal loss=95.28%
