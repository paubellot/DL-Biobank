"""
Generic setup of the data sources and the model training.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
and also on
    https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

"""

# import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from scipy.stats import pearsonr
import logging

# Helper: Early stopping.
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=2, verbose=0, mode='auto')
# In case that your training loss is not dropping - which means you are learning nothing after each epoch.
# It look like there's nothing to learn in this model, aside from some trivial linear-like fit or cutoff value.



def compile_model_mlp(geneparam, input_shape):
    """Compile a sequential model.

    Args:
        geneparam (dict): the parameters of the network

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
    print("Architecture:%d,%s,%s,%d,%.2f%%,%.2f%%" % (nb_neurons, activation, optimizer,
                                                             nb_layers, dropout,weight_decay))
   
    logging.info("Architecture:%d,%s,%s,%d,%.2f%%,%.2f%%" % (nb_neurons, activation, optimizer,
                                                             nb_layers, dropout, weight_decay))

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            if weight_decay>0:
                model.add(Dense(nb_neurons, activation=activation, input_dim=input_shape,
                                kernel_regularizer=l2(weight_decay)))
            else:
                model.add(Dense(nb_neurons, activation=activation, input_dim=input_shape))

        else:
            if weight_decay > 0:
                model.add(Dense(nb_neurons, activation=activation,  kernel_regularizer=l2(weight_decay)))
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


def compile_model_cnn(geneparam, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        geneparam (dict): the parameters of the genome

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = geneparam['nb_layers']
    nb_neurons = geneparam['nb_neurons']
    activation = geneparam['activation']
    optimizer = geneparam['optimizer']

    logging.info("Architecture:%d,%s,%s,%d" % (nb_neurons, activation, optimizer, nb_layers))

    model = Sequential()

    # Add each layer.
    for i in range(0, nb_layers):
        # Need input shape for first layer.
        if i == 0:
            model.add(
                Conv2D(nb_neurons, kernel_size=(3, 3), activation=activation, padding='same', input_shape=input_shape))
        else:
            model.add(Conv2D(nb_neurons, kernel_size=(3, 3), activation=activation))

        if i < 2:  # otherwise we hit zero
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(nb_neurons, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    # BAYESIAN CONVOLUTIONAL NEURAL NETWORKS WITH BERNOULLI APPROXIMATE VARIATIONAL INFERENCE
    # need to read this paper

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def get_data(dataset):
    from utils.utils import clean_data
    from sklearn.model_selection import train_test_split

    markers, pheno = clean_data(dataset.trait, dataset.k)
    x_train, x_test, y_train, y_test = train_test_split(markers, pheno, test_size=0.33, random_state=42)
    return (markers.shape[1], x_train, x_test, y_train, y_test)


def train_and_score(geneparam, dataset):
    """Train the model, return test loss.

    Args:
        geneparam (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    logging.info("Getting datasets")
    input_shape, x_train, x_test, y_train, y_test = get_data(dataset)
    logging.info("Compling Keras model")
    model = compile_model_mlp(geneparam, input_shape)

    history = LossHistory()

    model.fit(x_train, y_train,
              epochs=1200,
              # using early stopping so no real limit - don't want to waste time on horrible architectures
              verbose=1,
              validation_data =(x_test, y_test),
              # callbacks=[history])
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test mse:', score[0])
    print('Test mae:', score[1])
    r = pearsonr(model.predict(x_test).ravel(), y_test)[0]
    print('Test r:', r)
    logging.info("R: %.3f" % r)
    K.clear_session()
    # we do not care about keeping any of this in memory -
    # we just need to know the final scores and the architecture

    if r != r:
        r = -1.0
    return r