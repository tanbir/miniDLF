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
### MNIST FNN
    from miniDLF.models import Sequential
    from miniDLF.layers import Dense, Dropout, Activation
    from miniDLF.optimizers import Adam
    from miniDLF.datasets import MNIST

    mnist = MNIST('./data/MNIST/mnist.pkl.gz', n_classes=10, input_shape=(784,))

    m = Sequential()
    m.add(Dense(300, input_shape=(784,)))
    m.add(Activation('leaky_relu'))
    m.add(Dropout(0.25))
    m.add(Dense(300))
    m.add(Activation('leaky_relu'))
    m.add(Dropout(0.25))
    m.add(Dense(300))
    m.add(Activation('leaky_relu'))
    m.add(Dropout(0.25))
    m.add(Dense(10))
    m.add(Activation('softmax'))
    m.compile(loss='cce', optimizer=Adam())
    m.summary()

    m.fit(dataset=mnist, epochs=25, minibatch_size = 256, early_stop_after = 10)
 
#### CPU output 
    Dense      :   input_shape =  [784]  output_shape =  (300,)  trainable parameters =  235500
    Activation :   input_shape =  (300,)  output_shape =  (300,)  trainable parameters =  0
    Dropout    :   input_shape =  (300,)  output_shape =  (300,)  trainable parameters =  0
    Dense      :   input_shape =  (300,)  output_shape =  (300,)  trainable parameters =  90300
    Activation :   input_shape =  (300,)  output_shape =  (300,)  trainable parameters =  0
    Dropout    :   input_shape =  (300,)  output_shape =  (300,)  trainable parameters =  0
    Dense      :   input_shape =  (300,)  output_shape =  (300,)  trainable parameters =  90300
    Activation :   input_shape =  (300,)  output_shape =  (300,)  trainable parameters =  0
    Dropout    :   input_shape =  (300,)  output_shape =  (300,)  trainable parameters =  0
    Dense      :   input_shape =  (300,)  output_shape =  (10,)  trainable parameters =  3010
    Activation :   input_shape =  (10,)  output_shape =  (10,)  trainable parameters =  0
    Total # trainable parameters: 419110
    

    Epoch 01: ==========================================================> loss: 0.38612 test_acc = 94.92% time: 5.79s
    Epoch 02: ==========================================================> loss: 0.12230 test_acc = 96.25% time: 5.33s
    Epoch 03: ==========================================================> loss: 0.08001 test_acc = 96.98% time: 5.17s
    Epoch 04: ==========================================================> loss: 0.05600 test_acc = 97.00% time: 5.14s
    Epoch 05: ==========================================================> loss: 0.03925 test_acc = 97.41% time: 5.24s
    Epoch 06: ==========================================================> loss: 0.02801 test_acc = 97.45% time: 5.22s
    Epoch 07: ==========================================================> loss: 0.02169 test_acc = 97.52% time: 5.17s
    Epoch 08: ==========================================================> loss: 0.01591 test_acc = 97.74% time: 5.24s
    Epoch 09: ==========================================================> loss: 0.01175 test_acc = 97.71% time: 5.19s
    Epoch 10: ==========================================================> loss: 0.01148 test_acc = 97.74% time: 5.14s
    Epoch 11: ==========================================================> loss: 0.00724 test_acc = 97.72% time: 5.23s
    Epoch 12: ==========================================================> loss: 0.00622 test_acc = 97.87% time: 5.20s
    Epoch 13: ==========================================================> loss: 0.00697 test_acc = 97.76% time: 5.40s
    Epoch 14: ==========================================================> loss: 0.00939 test_acc = 97.33% time: 5.39s
    Epoch 15: ==========================================================> loss: 0.01766 test_acc = 97.26% time: 5.38s
    Epoch 16: ==========================================================> loss: 0.01680 test_acc = 97.50% time: 5.26s
    Epoch 17: ==========================================================> loss: 0.00881 test_acc = 97.90% time: 5.44s
    Epoch 18: ==========================================================> loss: 0.00708 test_acc = 97.98% time: 5.64s
    Epoch 19: ==========================================================> loss: 0.00424 test_acc = 98.05% time: 5.45s
    Epoch 20: ==========================================================> loss: 0.00168 test_acc = 98.10% time: 5.42s
    Accuracy: Maximum=98.10%; With optimal loss=98.10%
 
