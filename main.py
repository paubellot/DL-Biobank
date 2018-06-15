import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr as r
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1, l2
sys.path.insert(0, os.path.join(os.path.expanduser("~"), 'Code/genomic_cnn/'))
from utils.utils import retrieve_data
from evolve_cnn.train import compile_model_cnn
from evolve.train import compile_model_mlp
from utils.utils import convert_to_individual_alleles

early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=2, verbose=0, mode='auto')


def CNN(traits=['height', 'BMI', 'WHR', 'BHMD', 'SBP'], verbose=0, unif=False, nbsnps=10000, p=None, reps=1):
    #cnn1
    param = [{'optimizer': 'nadam', 'size_window': 2, 'activation': 'softplus', 'nb_neurons': 64, 'stride': 'one',
              'nb_cnn_layers': 1, 'filters': 16, 'weight_decay': 0.0, 'nb_layers': 3,
              'dropout': 0.01, 'batch_norm': True}]
    #cnn2
    param.append({'optimizer': 'nadam', 'size_window': 2, 'activation': 'elu', 'nb_neurons': 32, 'stride': 'one',
                  'nb_cnn_layers': 1, 'filters': 32, 'weight_decay': 0.0, 'nb_layers': 3,
                  'dropout': 0.01, 'batch_norm': False})
    #cnn3
    param.append({'optimizer': 'rmsprop', 'size_window': 3, 'activation': 'linear', 'nb_neurons': 32, 'stride': 'one',
                  'nb_cnn_layers': 1, 'filters': 16, 'weight_decay': 0.0, 'nb_layers': 1,
                  'dropout': 0.01, 'batch_norm': False})
    R = {}
    for t in traits:
        best = 0
        print(t)
        x_tr, x_tst, y_tr, y_tst = retrieve_data(t, nbsnps, unif=unif)
        x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=0.33)
        n_snps = x_tr.shape[1]
        x_tr = np.expand_dims(x_tr, axis=2)
        x_val = np.expand_dims(x_val, axis=2)
        x_tst = np.expand_dims(x_tst, axis=2)
        f = os.path.join(os.path.expanduser("~"), 'Code/genomic_cnn/models', "Model_" + t + "_cnn_"
                         + str(n_snps / 1000) + "k" + ("_unif" if unif else "_best") + ".h5")
        n = 0
        if p is None:
            res = np.zeros((len(param), 2))
            for g in param:
                print(g)
                for x in range(0, reps):
                    m = compile_model_cnn(g, (n_snps, 1))
                    m.fit(x_tr, y_tr, epochs=1200, verbose=verbose, validation_data=(x_val, y_val),
                          callbacks = [early_stopper])
                    if r(m.predict(x_val).ravel(), y_val)[0] > res[n, 0]:
                        print(r(m.predict(x_val).ravel(), y_val)[0])
                        print(x)
                        res[n, 0] = r(m.predict(x_val).ravel(), y_val)[0]
                        res[n, 1] = r(m.predict(x_tst).ravel(), y_tst)[0]
                    if res[n, 0] > best:
                        print("A better network was found with r: %.3f" % res[n, 0])
                        print(g)
                        m.save(f)
                        best = res[n, 0]
                n = n + 1
        else:
            res = np.zeros((reps, 2))
            g = param[p]
            for i in range(0, reps):
                m = compile_model_cnn(g, (n_snps, 1))
                m.fit(x_tr, y_tr, epochs=1200, verbose=verbose, validation_data=(x_val, y_val),callbacks=[early_stopper])
                res[i, :] = (r(m.predict(x_val).ravel(), y_val)[0], r(m.predict(x_tst).ravel(), y_tst)[0])
        R[t+"_tr"] = res[:, 0]
        R[t+"_tst"] = res[:, 1]
    print(pd.DataFrame(R).to_csv(float_format='%.3f', index=False))
    logging.info(pd.DataFrame(R).to_csv(float_format='%.3f', index=False))


def MLP(traits=['height', 'BMI', 'WHR', 'BHMD', 'SBP'], verbose=0, unif=False, nbsnps=10000, p=None, reps=1,hot=False):


    #mlp1
    geneparam = [{'optimizer': 'rmsprop', 'activation': 'elu', 'nb_neurons': 32,
                      'weight_decay': 0.01, 'nb_layers': 1, 'dropout': 0.02}]

    # mlp2
    geneparam.append({'optimizer': 'adagrad', 'activation': 'elu', 'nb_neurons': 64, 'weight_decay': 0.01,
                      'nb_layers': 2, 'dropout': 0.03})
    # mlp3
    geneparam.append({'optimizer': 'adam', 'activation': 'softplus', 'nb_neurons': 32,
                      'weight_decay': 0.01, 'nb_layers': 5, 'dropout': 0.02})

    R = {}
    for t in traits:
        print(t)
        best = 0
        x_tr, x_tst, y_tr, y_tst = retrieve_data(t, nbsnps, unif=unif)
        x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=0.33)
        if hot:
            x_tr = convert_to_individual_alleles(x_tr)
            x_val = convert_to_individual_alleles(x_val)
            x_tst = convert_to_individual_alleles(x_tst)
            n_snps = x_tr.shape[1]
            f = os.path.join(os.path.expanduser("~"), 'Code/genomic_cnn/models',
                             "Model_" + t + "_mlp_" + str(n_snps / 1000) \
                             + "kHot" + ("_unif" if unif else "_best") + ".h5")
        else:
            n_snps = x_tr.shape[1]
            f = os.path.join(os.path.expanduser("~"), 'Code/genomic_cnn/models', "Model_" + t + "_mlp_"
                             + str(n_snps / 1000)  + "k" + ("_unif" if unif else "_best") + ".h5")
        n = 0
        if p is None:
            res = np.zeros((len(geneparam), 2))
            for g in geneparam:
                print(g)
                for x in range(0, reps):
                    m = compile_model_mlp(g, n_snps)
                    m.fit(x_tr, y_tr, epochs=1200, validation_data=(x_val, y_val), callbacks=[early_stopper], verbose=verbose)
                    if r(m.predict(x_val).ravel(), y_val)[0] > res[n, 0]:
                        print(r(m.predict(x_val).ravel(), y_val)[0])
                        print(x)
                        res[n, 0] = r(m.predict(x_val).ravel(), y_val)[0]
                        res[n, 1] = r(m.predict(x_tst).ravel(), y_tst)[0]
                    if res[n, 0] > best:
                        print("A better network was found with r: %.3f" % res[n,0])
                        print(g)
                        m.save(f)
                        best = res[n, 0]
                K.clear_session()
                n = n + 1
        else:
            res = np.zeros((reps, 2))
            g = geneparam[p]
            for i in range(0, reps):
                m = compile_model_mlp(g, n_snps)
                m.fit(x_tr, y_tr, epochs=1200, verbose=verbose, validation_data=(x_val, y_val),
                      callbacks=[early_stopper])
                res[i, :] = (r(m.predict(x_val).ravel(), y_val)[0], r(m.predict(x_tst).ravel(), y_tst)[0])
        R[t + "_tr"] = res[:, 0]
        R[t + "_tst"] = res[:, 1]
    print(pd.DataFrame(R).to_csv(float_format='%.3f', index=False))
    logging.info(pd.DataFrame(R).to_csv(float_format='%.3f', index=False))



def lin_models(lasso=True, traits=['height', 'BMI', 'WHR', 'BHMD', 'SBP'], nbsnps=10000,verbose=0, hot=False, unif=False, reps=1):
    alpha = [0.01]
    R = {}
    for t in traits:
        print(t)
        x_tr, x_tst, y_tr, y_tst = retrieve_data(t, nbsnps, unif=unif)
        x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=0.33)
        if hot:
            x_tr = convert_to_individual_alleles(x_tr)
            x_val = convert_to_individual_alleles(x_val)
            x_tst = convert_to_individual_alleles(x_tst)
        nb_snps = x_tr.shape[1]
        res = np.zeros((len(alpha), 3))
        n = 0
        for a in alpha:
            print(a)
            for i in range(0,reps):
                m = Sequential()
                if lasso:
                    m.add(Dense(1, input_dim=nb_snps,kernel_regularizer=l1(a)))
                else:
                    m.add(Dense(1, input_dim=nb_snps, kernel_regularizer=l2(a)))
                m.compile(loss='mse', optimizer='adam')
                m.fit(x_tr, y_tr, epochs=1000, callbacks=[EarlyStopping()], validation_data=(x_val, y_val), verbose=verbose)
                if r(m.predict(x_val).ravel(), y_val)[0] > res[n, 0]:
                    print(r(m.predict(x_val).ravel(), y_val)[0])
                    print(i)
                    res[n, 0] = r(m.predict(x_val).ravel(), y_val)[0]
                    res[n, 1] = r(m.predict(x_tst).ravel(), y_tst)[0]
                K.clear_session()
            print(res[n, 1])
            n = n+1
        R[t+"val"] = res[:, 0]
        R[t+"tst"] = res[:, 1]
    R["alpha"] = alpha
    print(pd.DataFrame(R).to_csv(float_format='%.3f', index=False))
    logging.info(pd.DataFrame(R).to_csv(float_format='%.3f', index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait", help="Trait to run optimization", default="height")
    parser.add_argument("-k", "--num_snps", help="Number of SNPs", default=10000, type=int)
    parser.add_argument("-v", "--verbose", help="Verbose", default=0, type=int)
    parser.add_argument("--unif", help="Use uniformly spaced spns", action='store_true')
    parser.add_argument("-f", "--file", help="filename", default=None)
    parser.add_argument("--hot", help="Use 1 hot encoding", action="store_true")
    parser.add_argument("-r", "--rep", help="Repetitions", default=1, type=int)
    parser.add_argument("-s","--specific", help="Train specific mlp/cnn model (int)", default=None, type=int)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--lasso', action='store_true')
    group.add_argument('--ridge', action='store_true')
    group.add_argument('--mlp', action='store_true')
    group.add_argument('--cnn', action='store_true')

    args = parser.parse_args()
    print args
    if args.lasso:
        method = "_lasso_"
    if args.ridge:
        method = "_ridge_"
    if args.mlp:
        method = "_mlp_"
    if args.cnn:
        method = "_cnn_"
    if args.hot:
        method = method + "1hot_"
    if args.file is None:
        filename = "Opt_" + args.trait + method + str(args.num_snps / 1000)
        if args.unif:
            filename = filename + "unifk.txt"
        else:
            filename = filename + "k.txt"
    else:
        filename = args.file
    print("logging to " + filename)
    # Setup logging.
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
        filename=filename
    )
    logging.info("***Evaluating %s for %s trait with %d snps***" % (method, args.trait, args.num_snps))
    logging.info(args)
    if args.lasso:
        lin_models(lasso=True, traits=[args.trait], nbsnps=args.num_snps,verbose=args.verbose,
                   hot=args.hot, unif=args.unif, reps=args.rep)
    if args.ridge:
        lin_models(lasso=False, traits=[args.trait], nbsnps=args.num_snps,verbose=args.verbose,
                   hot=args.hot, unif=args.unif, reps=args.rep)
    if args.mlp:
        MLP(traits=[args.trait], nbsnps=args.num_snps, verbose=args.verbose, hot=args.hot, unif=args.unif,
            reps=args.rep, p=args.specific)
    if args.cnn:
        CNN(traits=[args.trait], nbsnps=args.num_snps,verbose=args.verbose,unif=args.unif,
            reps=args.rep, p=args.specific)
