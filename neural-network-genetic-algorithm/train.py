"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
import sys
sys.path.insert(0, '../')
import Utils
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_diagram import ascii
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)
reduce_lr = ReduceLROnPlateau(factor=0.2, verbose=0, patience=2, min_lr=1e-5)

def get_cifar10():
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 64
    input_shape = (3072,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)
    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def get_mnist():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 128
    input_shape = (784,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)


def get_boston():
    from sklearn.datasets import load_boston
    batch_size = 32
    nb_classes = 1
    boston = load_boston()
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(boston['data'],boston['target'])
    input_shape = (x_train.shape[1],)
    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)


def get_mg():
    batch_size = 32
    nb_classes = 1
    x_train, x_test, y_train, y_test = Utils.load_data(small=False)
    input_shape = (x_train.shape[1],)
    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)


def get_10k():
    batch_size = 32
    nb_classes = 1
    data = np.load('/home/pau/Code/biobank/notebooks/data/gwas_10k.npz')
    x_train = data['tr']
    x_test = data['tst']
    y_train = data['ytr']
    y_test = data['ytst']
    input_shape = (x_train.shape[1],)
    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)


def compile_model(net, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    network = net.network
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    if net.reg:
        model.add(Dense(nb_classes))
        model.compile(loss='mse', optimizer=optimizer,
                      metrics=['mae'])
    else:
        model.add(Dense(nb_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def Eval(x, y, model):
    pred = model.predict(x).ravel()
    cor = np.corrcoef(y, pred)[0, 1]
    return cor**2


def train_and_score(network, dataset, store_val, store_tst):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    if dataset == 'cifar10':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_cifar10()
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_mnist()
    elif dataset == "mg":
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_mg()
    elif dataset == "boston":
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_boston()
    elif dataset == "10k":
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_10k()

    xtr, xval, ytr, yval = train_test_split(x_train,y_train)
    model = compile_model(network, nb_classes, input_shape)
    print(model.summary())
    model.fit(xtr, ytr,
              batch_size=batch_size,
              epochs=10000,  # using early stopping, so no real limit
              verbose=0,
              validation_data = (xval, yval),
              callbacks=[early_stopper, reduce_lr])
    df_val = store_val['df']
    df_val = df_val.append({"nb_layers": network.network['nb_layers'],
                    "nb_neurons": network.network['nb_neurons'],
                    "activation": network.network['activation'],
                    "optimizer": network.network['optimizer'],
                    "r2": Eval(xval, yval, model)
                    }, ignore_index=True)
    print(df_val.tail())
    store_val['df'] = df_val
    store_val.flush()
    score = model.evaluate(xval, yval, verbose=0)

    df_tst = store_tst['df']
    df_tst = df_tst.append({"nb_layers": network.network['nb_layers'],
                            "nb_neurons": network.network['nb_neurons'],
                            "activation": network.network['activation'],
                            "optimizer": network.network['optimizer'],
                            "r2": Eval(x_test, y_test, model)
                            }, ignore_index=True)
    print(df_tst.tail())
    store_tst['df'] = df_tst
    store_tst.flush()

    network.model = ascii(model)
    return -score[0]  # 1 is accuracy. 0 is loss.