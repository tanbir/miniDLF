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
    

    Epoch 01: ==============================================> loss: 1.44277 train_acc = 87.20% test_acc = 87.84% time: 17.04s
    Epoch 02: ==============================================> loss: 0.34213 train_acc = 91.89% test_acc = 91.64% time: 16.44s
    Epoch 03: ==============================================> loss: 0.24196 train_acc = 93.91% test_acc = 93.59% time: 16.92s
    Epoch 04: ==============================================> loss: 0.18950 train_acc = 95.05% test_acc = 94.46% time: 16.24s
    Epoch 05: ==============================================> loss: 0.16093 train_acc = 95.70% test_acc = 94.58% time: 16.02s
    Epoch 06: ==============================================> loss: 0.13941 train_acc = 96.21% test_acc = 95.59% time: 15.56s
    Epoch 07: ==============================================> loss: 0.12213 train_acc = 96.60% test_acc = 95.78% time: 16.84s
    Epoch 08: ==============================================> loss: 0.11234 train_acc = 96.87% test_acc = 96.05% time: 15.47s
    Epoch 09: ==============================================> loss: 0.09811 train_acc = 97.05% test_acc = 96.19% time: 15.42s
    Epoch 10: ==============================================> loss: 0.09065 train_acc = 97.50% test_acc = 96.25% time: 16.92s
    Epoch 11: ==============================================> loss: 0.08184 train_acc = 97.84% test_acc = 96.53% time: 15.52s
    Epoch 12: ==============================================> loss: 0.07784 train_acc = 97.91% test_acc = 96.66% time: 16.70s
    Epoch 13: ==============================================> loss: 0.07036 train_acc = 98.00% test_acc = 96.59% time: 15.77s
    Epoch 14: ==============================================> loss: 0.06715 train_acc = 98.09% test_acc = 96.84% time: 16.36s
    Epoch 15: ==============================================> loss: 0.06497 train_acc = 98.26% test_acc = 96.69% time: 16.51s
    Epoch 16: ==============================================> loss: 0.05742 train_acc = 98.22% test_acc = 96.93% time: 15.68s
    Epoch 17: ==============================================> loss: 0.05432 train_acc = 98.38% test_acc = 97.10% time: 15.73s
    Epoch 18: ==============================================> loss: 0.05422 train_acc = 98.41% test_acc = 97.00% time: 17.20s
    Epoch 19: ==============================================> loss: 0.04645 train_acc = 98.61% test_acc = 97.13% time: 16.68s
    Epoch 20: ==============================================> loss: 0.04794 train_acc = 98.64% test_acc = 97.20% time: 15.71s
    Epoch 21: ==============================================> loss: 0.04664 train_acc = 98.67% test_acc = 97.16% time: 16.68s
    Epoch 22: ==============================================> loss: 0.04434 train_acc = 98.77% test_acc = 97.17% time: 16.76s
    Epoch 23: ==============================================> loss: 0.04340 train_acc = 98.82% test_acc = 97.27% time: 15.31s
    Epoch 24: ==============================================> loss: 0.04187 train_acc = 98.84% test_acc = 97.24% time: 15.67s
    Epoch 25: ==============================================> loss: 0.04020 train_acc = 98.89% test_acc = 97.09% time: 15.12s
    Accuracy: Maximum=97.27%; With optimal loss=97.09%
 
