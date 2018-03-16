import sys
sys.path.insert(0, '/home/pau/Code/biobank/')
import Utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1, l2
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def data_rand(k=50000):
    snp_list = np.random.permutation(500000)[0:k]
    x_train, _, _ = Utils.readgbinfile(fdir + 'genosTRN.bin', snp_list=snp_list, std=False)
    x_test, _, _ = Utils.readgbinfile(fdir + 'genosTST.bin', snp_list=snp_list, std=False)
    _, _, y_train, y_test = Utils.load_data(small=False)
    return x_train, x_test, y_train, y_test


def Eval(x, y, model):
    pred = model.predict(x).ravel()
    cor = np.corrcoef(y, pred)[0, 1]**2
    return cor


def model1(x_train,y_train,v=1):
    model = Sequential()
    model.add(Dense(256, input_dim=x_train.shape[1], activation="sigmoid"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="rmsprop",loss="mse",metrics=['mae'])
    reduce_lr = ReduceLROnPlateau(factor=0.2, verbose=2, patience=2, min_lr=1e-5)
    early_stopping = EarlyStopping(patience=4, verbose=2)
    callbacks = [reduce_lr, early_stopping]
    model.fit(x_train, y_train, callbacks=callbacks, epochs=150, validation_split=0.2, verbose=v)
    return model


def model2(x_train,y_train,v=1):
    model = Sequential()
    model.add(Dense(128, input_dim=x_train.shape[1], activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="rmsprop",loss="mse",metrics=['mae'])
    reduce_lr = ReduceLROnPlateau(factor=0.2, verbose=2, patience=2, min_lr=1e-5)
    early_stopping = EarlyStopping(patience=4, verbose=2)
    callbacks = [reduce_lr, early_stopping]
    model.fit(x_train, y_train, callbacks=callbacks, epochs=150, validation_split=0.2, verbose=v)
    return model


def lasso(x_train,y_train,a=0.001,v=1):
    model = Sequential()
    model.add(Dense(1,input_dim=x_train.shape[1],kernel_regularizer=l1(a)))
    model.compile(optimizer="rmsprop",loss="mse",metrics=['mae'])
    reduce_lr = ReduceLROnPlateau(factor=0.2, verbose=2, patience=2, min_lr=1e-5)
    early_stopping = EarlyStopping(patience=4, verbose=2)
    callbacks = [reduce_lr,early_stopping]
    model.fit(x_train,y_train,callbacks=callbacks,epochs=150,validation_split=0.2,verbose=v)
    return model


def ridge(x_train,y_train,a=0.005,v=1):
    model = Sequential()
    model.add(Dense(1,input_dim=x_train.shape[1],kernel_regularizer=l2(a)))
    model.compile(optimizer="rmsprop",loss="mse",metrics=['mae'])
    reduce_lr = ReduceLROnPlateau(factor=0.2, verbose=2, patience=2, min_lr=1e-5)
    early_stopping = EarlyStopping(patience=4, verbose=2)
    callbacks=[reduce_lr,early_stopping]
    model.fit(x_train,y_train,callbacks=callbacks,epochs=150,validation_split=0.2,verbose=v)
    return model


def Train(x_train,x_test,y_train,y_test, model):
    if model == "mlp1":
        m = model1(x_train, y_train,v=0)
    elif model == "mlp2":
        m = model2(x_train, y_train, v=0)
    elif model == "lasso":
        m = lasso(x_train, y_train, v=0)
    elif model == "ridge":
        m = ridge(x_train, y_train, v=0)

    return Eval(x_test,y_test,m)
