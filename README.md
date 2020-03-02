# miniDLF 
(Mini Deep Learning Framework)

# Introduction

This is a humble attempt (inspired by [Stanford CS class CS231n](http://cs231n.github.io/)) to implement a simple Numpy-based <a hraf=https://keras.io/ target=blank>Keras</a>-like framework for designing Neural Networks. This is not intended to be used for your production environment. But feel free to use it for learning and development. Plenty of room for code-optimization.  

# Contents 

## Network architectures
* Feedforward Neural Network 
* Convolutional Neural Network 
* Recurrent Neural Network
* Gated Recurrent Units (GRU)
* Long Short Term Memomry Units (LSTM)

## Layers
### Core layers
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

### Recurrent layers
| Layer                    | Syntex to create an object                                                       |
|:-------------------------|:---------------------------------------------------------------------------------|
| RNN                      | `RNN(n_units, input_shape = None)`                                  |
| LSTM                     | `LSTM(n_units, input_shape = None)`                                              |
| GRU                      | `GRU(n_units, input_shape = None)`                                               |

### Activations
| Layer                    | Syntex to create an object                                                       |
|:-------------------------|:---------------------------------------------------------------------------------|
| Activation               | `Activation(activation, alpha=0.0001, max_value=2.5, scale=1.0)`                 |
| ReLU (activation)        | `ReLU(alpha=0.0, max_value=np.Infinity)`                                         |
| Leaky-ReLU (activation)  | `ReLU(alpha, max_value)`                                                         |
| ELU (activation)         | `ELU(alpha, max_value)`                                                          |
| Sigmoid (activation)     | `Sigmoid()`                                                                      |
| Tanh (activation)        | `Tanh()`                                                                         |
| Softmax (activation)     | `Softmax()`                                                                      |

## Optimization algorithms
| Optimizer | Syntex to create an object                                              |
|:----------|:------------------------------------------------------------------------|
| SGD       | `SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)`                 |
| Adagrad   | `Adagrad(lr=0.01, decay=1e-6, epsilon=1e-7)`                            |
| RMSProp   | `RMSProp(lr=0.001, decay=1e-6, rho=0.9, epsilon=1e-6)`                  |
| Adam      | `Adam(lr=0.001, decay=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-6)`    |
| Nadam     | `Nadam(lr=0.001, decay=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-6)`   |
| AMSGrad   | `AMSGrad(lr=0.001, decay=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-6)` |

## Model function: Sequential
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

## Some example runs
### Multilayer Perceptron using MNIST
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

  Epoch 01: =======================================> loss: 2.03288 train_acc = 87.32% test_acc = 87.98% time: 26.75s
  Epoch 02: =======================================> loss: 0.36526 train_acc = 91.98% test_acc = 92.62% time: 25.25s
  Epoch 03: =======================================> loss: 0.23474 train_acc = 94.33% test_acc = 94.21% time: 24.82s
  Epoch 04: =======================================> loss: 0.18120 train_acc = 94.99% test_acc = 95.02% time: 24.28s
  Epoch 05: =======================================> loss: 0.15364 train_acc = 95.96% test_acc = 95.63% time: 23.91s
  Epoch 06: =======================================> loss: 0.12797 train_acc = 96.29% test_acc = 95.68% time: 24.35s
  Epoch 07: =======================================> loss: 0.11399 train_acc = 96.94% test_acc = 96.39% time: 23.33s
  Epoch 08: =======================================> loss: 0.09870 train_acc = 97.26% test_acc = 96.37% time: 23.97s
  Epoch 09: =======================================> loss: 0.09071 train_acc = 97.59% test_acc = 96.72% time: 26.06s
  Epoch 10: =======================================> loss: 0.07748 train_acc = 97.75% test_acc = 96.95% time: 23.73s
  Epoch 11: =======================================> loss: 0.07103 train_acc = 97.93% test_acc = 96.80% time: 24.08s
  Epoch 12: =======================================> loss: 0.06592 train_acc = 98.11% test_acc = 96.98% time: 24.05s
  Epoch 13: =======================================> loss: 0.05985 train_acc = 98.35% test_acc = 96.75% time: 26.55s
  Epoch 14: =======================================> loss: 0.05286 train_acc = 98.33% test_acc = 97.13% time: 26.24s
  Epoch 15: =======================================> loss: 0.04977 train_acc = 98.57% test_acc = 97.35% time: 23.85s
  Epoch 16: =======================================> loss: 0.04704 train_acc = 98.54% test_acc = 97.26% time: 24.55s
  Epoch 17: =======================================> loss: 0.04373 train_acc = 98.69% test_acc = 97.20% time: 23.78s
  Epoch 18: =======================================> loss: 0.04225 train_acc = 98.77% test_acc = 97.29% time: 23.55s
  Epoch 19: =======================================> loss: 0.03697 train_acc = 98.97% test_acc = 97.64% time: 22.62s
  Epoch 20: =======================================> loss: 0.03574 train_acc = 98.92% test_acc = 97.32% time: 24.08s
  Epoch 21: =======================================> loss: 0.03627 train_acc = 99.01% test_acc = 97.50% time: 23.51s
  Epoch 22: =======================================> loss: 0.03262 train_acc = 99.00% test_acc = 97.39% time: 23.57s
  Epoch 23: =======================================> loss: 0.03035 train_acc = 99.10% test_acc = 97.55% time: 23.21s
  Epoch 24: =======================================> loss: 0.03081 train_acc = 99.05% test_acc = 97.36% time: 23.40s
  Epoch 25: =======================================> loss: 0.02681 train_acc = 99.09% test_acc = 97.44% time: 23.03s
  Accuracy: Maximum=97.64%; With optimal loss=97.44%
 
