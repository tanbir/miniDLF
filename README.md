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
