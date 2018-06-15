"""
Generic setup of the data sources and the model training.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
and also on
    https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

"""

# import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, Callback
from keras import backend as K
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import logging
from utils.bb_data import DataDefinition
from sklearn.model_selection import train_test_split

# Helper: Early stopping.
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=2, verbose=0, mode='auto')


# In case that your training loss is not dropping - which means you are learning nothing after each epoch.
# It look like there's nothing to learn in this model, aside from some trivial linear-like fit or cutoff value.

def compile_model_cnn(geneparam, input_shape):
    """Compile a sequential model.

    Args:
        geneparam (dict): the parameters of the network
            geneparam = {
            'nb_neurons': [16, 32, 64, 128],
            'nb_layers': [1, 2, 3],
            'nb_cnn_layers': [1, 2, 3],
            'batch_norm': [True,False],
            'activation': ['relu', 'elu', 'softplus', 'linear'],
            'optimizer': ['rmsprop', 'nadam'],
            'dropout': [0.,  0.075],
            'filters': [16, 32, 64, 128],
            'size_window': [2,3,5,10],
            'stride' : ["equal","one"],
            'weight_decay': [0., 0.075]
         }
    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = geneparam['nb_layers']
    nb_neurons = geneparam['nb_neurons']
    activation = geneparam['activation']
    optimizer = geneparam['optimizer']
    dropout = geneparam['dropout']
    weight_decay = geneparam['weight_decay']
    nb_cnn_layers = geneparam['nb_cnn_layers']
    batch_norm = geneparam['batch_norm']
    filters = geneparam['filters']
    size_window  = geneparam['size_window']
    stride = geneparam['stride']
    print("Architecture:[%d,%d,%s]*%d,bn=%s;%d,%s,%s,%d,dr=%.2f,wd=%.2f" % (size_window, filters, stride, nb_cnn_layers,
                                                                            batch_norm, nb_neurons, activation,
                                                                            optimizer, nb_layers,
                                                                            dropout, weight_decay))

    logging.info("Architecture:[%d,%d,%s]*%d,bn=%s;%d,%s,%s,%d,dr=%.2f,wd=%.2f" % (size_window, filters, stride,
                                                                                   nb_cnn_layers, batch_norm,
                                                                                   nb_neurons, activation,
                                                                                   optimizer, nb_layers,
                                                                                   dropout, weight_decay))
    model = Sequential()
    if stride == "equal":
        st = size_window
    else:
        st = 1
    # Add each layer.
    for i in range(nb_cnn_layers):
        if i == 0:
            if weight_decay > 0:
                model.add(Conv1D(filters=filters, kernel_size=size_window, strides=st,
                                 activation=activation, input_shape=input_shape,
                                 kernel_regularizer=l2(weight_decay)))
            else:
                model.add(Conv1D(filters=filters, kernel_size=size_window, strides=st,
                                 activation=activation, input_shape=input_shape))
        else:
            if weight_decay > 0:
                model.add(Conv1D(filters=filters, kernel_size=size_window, strides=st,
                                 activation=activation, kernel_regularizer=l2(weight_decay)))
            else:
                model.add(Conv1D(filters=filters, kernel_size=size_window, strides=st,activation=activation))
        if batch_norm:
            model.add(BatchNormalization())
        if dropout > 0:
            model.add(Dropout(dropout))  # dropout for each layer
        model.add(MaxPooling1D())
    model.add(Flatten())
    for i in range(nb_layers):
        if weight_decay > 0:
            model.add(Dense(nb_neurons, activation=activation, kernel_regularizer=l2(weight_decay)))
        else:
            model.add(Dense(nb_neurons, activation=activation))
        if dropout > 0:
            model.add(Dropout(dropout))  # dropout for each layer

    # Output layer.
    model.add(Dense(1))
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def get_data(dataset):
    markers = data[dataset.trait][str(dataset.k)].markers_cnn_tr()
    pheno = data[dataset.trait][str(dataset.k)].pheno_tr()
    has_trait_data = pd.DataFrame(pheno).notnull().values.ravel()
    markers, pheno = markers[has_trait_data, :], pheno[has_trait_data]
    x_train, x_test, y_train, y_test = train_test_split(markers, pheno, test_size=0.33, random_state=42)
    return (markers.shape[1], x_train, x_test, y_train, y_test)


def retrieve_data(dataset):
    Ds = DataDefinition(dataset.trait,dataset.k)
    xtr = Ds.markers_cnn_tr()
    ytr = Ds.pheno_tr()
    has_trait_data = pd.DataFrame(ytr).notnull().values.ravel()
    xtr, ytr = xtr[has_trait_data, :], ytr[has_trait_data]
    xtst = Ds.markers_cnn_tst()
    ytst = Ds.pheno_tst()
    has_trait_data = pd.DataFrame(ytst).notnull().values.ravel()
    xtst, ytst = xtst[has_trait_data, :], ytst[has_trait_data]
    return (xtr, xtst, ytr, ytst)


def train_and_score(geneparam, dataset):
    """Train the model, return test loss.

    Args:
        geneparam (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    logging.info("Getting datasets")
    input_shape, x_train, x_test, y_train, y_test = get_data(dataset)
    train_data = np.expand_dims(x_train, axis=2)
    test_data = np.expand_dims(x_test, axis=2)
    input_shape = (input_shape, 1)
    logging.info("Compling Keras model")
    model = compile_model_cnn(geneparam, input_shape)

    history = LossHistory()

    model.fit(train_data, y_train,
              epochs=1200,
              # using early stopping so no real limit - don't want to waste time on horrible architectures
              verbose=1,
              validation_data=(test_data, y_test),
              # callbacks=[history])
              callbacks=[early_stopper])

    score = model.evaluate(test_data, y_test, verbose=0)

    print('Test mse:', score[0])
    print('Test mae:', score[1])
    r = pearsonr(model.predict(test_data).ravel(), y_test)[0]
    print("Test r:", r)
    K.clear_session()
    # we do not care about keeping any of this in memory -
    # we just need to know the final scores and the architecture
    if r != r:
        r = -1.0
    return r