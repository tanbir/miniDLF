# miniDLF 
(Mini Deep Learning Framework)

# 1 Introduction

This is a humble attempt (inspired by [Stanford CS class CS231n](http://cs231n.github.io/)) to implement a simple Numpy-based <a hraf=https://keras.io/ target=blank>Keras</a>-like framework for designing Neural Networks. This is not intended to be used for your production environment. But feel free to use it for learning and development. Plenty of room for code-optimization.  

# 2 Framework 

## 2.1 Network architectures
* Feedforward Neural Network 
* Convolutional Neural Network 
* Recurrent Neural Network
* Gated Recurrent Units (GRU)
* Long Short Term Memomry Units (LSTM)

## 2.2 Layers
### 2.2.1 Core layers
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

### 2.2.2 Recurrent layers
| Layer                    | Syntex to create an object                                                       |
|:-------------------------|:---------------------------------------------------------------------------------|
| RNN                      | `RNN(n_units, input_shape = None)`                                  |
| LSTM                     | `LSTM(n_units, input_shape = None)`                                              |
| GRU                      | `GRU(n_units, input_shape = None)`                                               |

### 2.2.3 Activations
| Layer                    | Syntex to create an object                                                       |
|:-------------------------|:---------------------------------------------------------------------------------|
| Activation               | `Activation(activation, alpha=0.0001, max_value=2.5, scale=1.0)`                 |
| ReLU (activation)        | `ReLU(alpha=0.0, max_value=np.Infinity)`                                         |
| Leaky-ReLU (activation)  | `ReLU(alpha, max_value)`                                                         |
| ELU (activation)         | `ELU(alpha, max_value)`                                                          |
| Sigmoid (activation)     | `Sigmoid()`                                                                      |
| Tanh (activation)        | `Tanh()`                                                                         |
| Softmax (activation)     | `Softmax()`                                                                      |

## 2.3 Optimization algorithms
| Optimizer | Syntex to create an object                                              |
|:----------|:------------------------------------------------------------------------|
| SGD       | `SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)`                 |
| Adagrad   | `Adagrad(lr=0.01, decay=1e-6, epsilon=1e-7)`                            |
| RMSProp   | `RMSProp(lr=0.001, decay=1e-6, rho=0.9, epsilon=1e-6)`                  |
| Adam      | `Adam(lr=0.001, decay=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-6)`    |
| Nadam     | `Nadam(lr=0.001, decay=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-6)`   |
| AMSGrad   | `AMSGrad(lr=0.001, decay=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-6)` |

## 2.4 Model function: Sequential
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

## 2.5 Some example runs
### 2.5.1 Multilayer Perceptron using MNIST
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
 
#### CPU output 
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

### 2.5.2 Convolutional Network using MNIST
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
 
#### CPU output 
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

### 2.5.3 Recurrent Neural Networks

    import numpy as np
    from miniDLF.models import Sequential
    from miniDLF.layers import RNN, GRU, LSTM, Activation
    from miniDLF.optimizers import Adam
    from miniDLF.datasets import TEXT2SEQ



    d = TEXT2SEQ('./data/TEXT/basic_rnn.txt', 50)  # extracted from wikipedia

    m = Sequential() 
    m.add(RNN(256, input_shape=d.input_shape))
    m.add(RNN(256, input_shape=d.input_shape))
    m.add(RNN(256, input_shape=d.input_shape))
    m.add(Activation('softmax'))
    m.compile(loss='cce', optimizer=Adam())
    m.summary()


    m.fit(dataset=d, epochs=1000, minibatch_size = 30, accuracy_threshold=0.96, early_stop_after = 30)
    
#### CPU output

    None :   input_shape =  [50 41]  output_shape =  [50 41]  trainable parameters =  86825
    None :   input_shape =  [50 41]  output_shape =  [50 41]  trainable parameters =  86825
    None :   input_shape =  [50 41]  output_shape =  [50 41]  trainable parameters =  86825
    Activation :   input_shape =  [50 41]  output_shape =  [50 41]  trainable parameters =  0
    Total # trainable parameters: 173650
    
    Epoch 01: ======================================> loss: 81.79636 train_acc = 39.56% test_acc = 39.58% time: 7.83s
    Epoch 02: ======================================> loss: 50.32352 train_acc = 62.82% test_acc = 62.54% time: 7.87s
    Epoch 03: ======================================> loss: 31.95252 train_acc = 77.79% test_acc = 77.02% time: 7.81s
    Epoch 04: ======================================> loss: 20.67803 train_acc = 88.05% test_acc = 87.56% time: 7.70s
    Epoch 05: ======================================> loss: 13.77051 train_acc = 92.24% test_acc = 91.77% time: 7.79s
    Epoch 06: ======================================> loss: 10.35432 train_acc = 93.55% test_acc = 93.02% time: 7.83s
    Epoch 07: ======================================> loss: 8.41647 train_acc = 94.13% test_acc = 93.36% time: 7.78s
    Epoch 08: ======================================> loss: 7.33525 train_acc = 94.71% test_acc = 93.88% time: 7.87s
    Epoch 09: ======================================> loss: 6.45532 train_acc = 95.05% test_acc = 93.98% time: 7.93s
    Epoch 10: ======================================> loss: 5.91990 train_acc = 95.31% test_acc = 94.01% time: 7.89s
    Epoch 11: ======================================> loss: 5.46156 train_acc = 95.39% test_acc = 94.19% time: 7.83s
    Epoch 12: ======================================> loss: 5.24236 train_acc = 95.49% test_acc = 94.22% time: 7.86s
    Epoch 13: ======================================> loss: 5.03817 train_acc = 95.70% test_acc = 94.31% time: 7.81s
    Epoch 14: ======================================> loss: 4.70586 train_acc = 95.81% test_acc = 94.31% time: 7.81s
    Epoch 15: ======================================> loss: 4.60796 train_acc = 95.82% test_acc = 94.33% time: 7.92s
    Epoch 16: ======================================> loss: 4.42895 train_acc = 95.88% test_acc = 94.46% time: 7.89s
    Epoch 17: ======================================> loss: 4.38104 train_acc = 95.97% test_acc = 94.52% time: 7.89s
    Epoch 18: ======================================> loss: 4.33359 train_acc = 95.93% test_acc = 94.21% time: 7.78s
    Epoch 19: ======================================> loss: 4.20446 train_acc = 96.00% test_acc = 94.42% time: 7.85s
    Epoch 20: ======================================> loss: 4.10394 train_acc = 96.06% test_acc = 94.26% time: 7.89s
    Terminating early (training accuracy threshold reached)
    Accuracy: Maximum=94.52%; With optimal loss=94.26%
    
