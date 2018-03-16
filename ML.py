from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1, l2
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def mlp1(x_train):
    model = Sequential()
    model.add(Dense(256, input_dim=x_train.shape[1], activation="sigmoid"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="rmsprop", loss="mse", metrics=['mae'])
    return model


def mlp2(x_train):
    model = Sequential()
    model.add(Dense(128, input_dim=x_train.shape[1], activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="rmsprop", loss="mse", metrics=['mae'])
    return model


def lasso(x_train, a=0.001):
    model = Sequential()
    model.add(Dense(1, input_dim=x_train.shape[1], kernel_regularizer=l1(a)))
    model.compile(optimizer="rmsprop", loss="mse", metrics=['mae'])
    return model


def ridge(x_train, a=0.005):
    model = Sequential()
    model.add(Dense(1, input_dim=x_train.shape[1], kernel_regularizer=l2(a)))
    model.compile(optimizer="rmsprop", loss="mse", metrics=['mae'])
    return model


def Eval(x, y, model):
    pred = model.predict(x).ravel()
    cor = np.corrcoef(y, pred)[0, 1] ** 2
    return cor


def Train(x_train, y_train, x_test, y_test, v=1):
    early_stopping = EarlyStopping(patience=4, verbose=0)
    reduce_lr = ReduceLROnPlateau(factor=0.2, verbose=0, patience=2, min_lr=1e-5)
    callbacks = [reduce_lr, early_stopping]
    m1 = mlp1(x_train)
    m1.fit(x_train, y_train, callbacks=callbacks, epochs=150, validation_split=0.2, verbose=v)
    m2 = mlp2(x_train)
    m2.fit(x_train, y_train, callbacks=callbacks, epochs=150, validation_split=0.2, verbose=v)
    l = lasso(x_train)
    l.fit(x_train, y_train, callbacks=callbacks, epochs=150, validation_split=0.2, verbose=v)
    r = lasso(x_train)
    r.fit(x_train, y_train, callbacks=callbacks, epochs=150, validation_split=0.2, verbose=v)
    return [Eval(x_test, y_test, m1), Eval(x_test, y_test, m2), Eval(x_test, y_test, l), Eval(x_test, y_test, r)]