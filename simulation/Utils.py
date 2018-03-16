import os
import numpy as np
import sys
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import pickle


ddir = '/home/data/biobank/'
wdir = "/home/pau/Data/BioB/"
dataset_file_sm = "snps_BioBank_small.pkl"
dataset_file = "snps_BioBank.pkl"
cldir = '/home/data/biobank/testdata/'
fdir = '/home/data/biobank/'



def read_gen_k(binfile, k=10000):
    import pandas as pd
    tb = pd.read_csv(ddir + "pvalue_height.txt", sep=" ")
    pvalues = tb['P-GWAS'].values
    snp_list = np.argsort(pvalues)[:k]
    print(snp_list)
    print(snp_list.shape)
    return read_gbinfile(binfile, snp_list=snp_list.tolist(), std=False)


def read_gen_pval(binfile, pval=None):
    if pval is None:
        with open(fdir + 'mengying.snp.list') as f:
            snp_list = [int(x.strip('\n')) for x in f.readlines()]
    else:
        import pandas as pd
        tb = pd.read_csv(ddir + "pvalue_height.txt", sep=" ")
        pvalues = tb['P-GWAS'].values
        snp_list = np.where(pvalues < pval)[0]
        print(snp_list)
        print(snp_list.shape)
    return read_gbinfile(binfile, snp_list=snp_list.tolist(), std=False)


def read_Binfile(binfile, snp_list=None, ind_list=None, std=True, maxval=None):
    """ reads genotype bin file into np array
            reads only samples and snps specified in snps, inds
            optionally scales X and set bounds
    """
    from sklearn import preprocessing
    import array
    nbyte = 1
    f = open(binfile, 'rb')
    n = int(struct.unpack('1d', f.read(8))[0])
    p = int(struct.unpack('1d', f.read(8))[0])
    sh = []
    if (snp_list is None):
        use_snp = np.ones(p, dtype=bool)
    else:
        use_snp = np.zeros(p, dtype=bool)
        use_snp[snp_list] = True
    if (ind_list is None):
        use_ind = np.ones(n, dtype=bool)
    else:
        use_ind = np.zeros(n, dtype=bool)
        use_ind[ind_list] = True
    # reads all genotypes for ith snp, and append only those needed
    sh.append(sum(use_ind))
    sh.append(sum(use_snp))
    X = np.zeros(sh).astype(np.float32)
    j = 0
    for i in range(p):
        # replace 'c' by 'd' if double precision
        line = f.read(nbyte * n)
        if (use_snp[i]):
            temp = np.asarray(array.array('b',line)).astype(np.float32)
            X[:, j] = temp[use_ind]
            j += 1
    if (std):
        X = preprocessing.scale(X)
        if (maxval is not None):
            X[X > maxval] = maxval
            X[X < -maxval] = -maxval
            X = preprocessing.scale(X, with_std=False)
    f.close()
    return X, X.shape[0], X.shape[1]



# This function reads genotype file, if present, snp_list specifies read snps, and ind_list individual list
#------------------------------------------------------------------------------
def readgbinfile(binfile, snp_list=None, ind_list=None, std=True, maxval=None):
#------------------------------------------------------------------------------
    """ reads genotype bin file into np array
        Optionally, reads only snps and samples specified in snp_list and ind_list
        by default (std=True) scales X
        optionally (maxval=MaxValue) set bounds
    """
    from sklearn import preprocessing
    import struct
    nbyte=1
    f = open(binfile, 'rb')
    n = int(struct.unpack('1d',f.read(8))[0])
    p = int(struct.unpack('1d',f.read(8))[0])
    if (snp_list is None):
        use_snp = np.ones(p,dtype=bool)
    else:
        use_snp = np.zeros(p,dtype=bool)
        use_snp[snp_list] = True
    if (ind_list is None):
        use_ind = np.ones(n,dtype=bool)
    else:
        use_ind = np.zeros(n,dtype=bool)
        use_ind[ind_list] = True
    # reads all genotypes for ith snp, and append only those needed
    X=[]
    for i in range(p):
        # replace 'c' by 'd' if double precision
        line = f.read(nbyte*n)
        if (use_snp[i]):
            temp = np.array(struct.unpack( str(n)+'b', line)).astype(np.float32)
            X.append(temp[use_ind])
    # X should be n x p
    X = np.array(X).T
    # optionally scales and set bounds
    if (std):
       X = preprocessing.scale(X)
       if (maxval is not None):
           X[X > maxval]  = maxval
           X[X < -maxval] = -maxval
           X = preprocessing.scale(X,with_std=False)
    f.close()
    return X,X.shape[0],X.shape[1]


# ------------------------------------------------------------------------------
def read_gbinfile(binfile, snp_list=None, ind_list=None, std=True, maxval=None):
    # ------------------------------------------------------------------------------
    """ reads genotype bin file into np array
        reads only samples and snps specified in snps, inds
        optionally scales X and set bounds
    """
    from sklearn import preprocessing
    import struct
    nbyte = 1
    f = open(binfile, 'rb')
    n = int(struct.unpack('1d', f.read(8))[0])
    p = int(struct.unpack('1d', f.read(8))[0])
    if (snp_list is None):
        use_snp = np.ones(p, dtype=bool)
    else:
        use_snp = np.zeros(p, dtype=bool)
        use_snp[snp_list] = True
    if (ind_list is None):
        use_ind = np.ones(n, dtype=bool)
    else:
        use_ind = np.zeros(n, dtype=bool)
        use_ind[ind_list] = True
    # reads all genotypes for ith snp, and append only those needed
    #X = []
    sh = []
    sh.append(sum(use_ind))
    sh.append(sum(use_snp))
    print(sh)
    j = 0
    X = np.zeros(sh).astype(np.float32)
    for i in range(p):
        # replace 'c' by 'd' if double precision
        line = f.read(nbyte * n)
        if (use_snp[i]):
            temp = np.array(struct.unpack(str(n) + 'b', line)).astype(np.float32)
            X[:, j] = temp[use_ind]
            j += 1
    # X should be n x p
    #X = np.array(X).T
    # optionally scales and set bounds
    if (std):
        X = preprocessing.scale(X)
        if (maxval is not None):
            X[X > maxval] = maxval
            X[X < -maxval] = -maxval
            X = preprocessing.scale(X, with_std=False)
    f.close()
    return X, X.shape[0], X.shape[1]


def Pd_read(yfile, name='height', header=True, ind_list=None, scale=True):
    import pandas as pd
    from sklearn import preprocessing
    tb = pd.read_csv(yfile)
    y = tb[name].values
    if (ind_list is not None):
        y = y[ind_list]
    if (scale):
        y = preprocessing.scale(y, with_std=True)
    return y


# -----------------------------------------------------------------------
def read_yfile(yfile, header=True, ind_list=None, std=True, maxval=None):
    # -----------------------------------------------------------------------
    """ reads y file into np array
        reads only samples specified in snps, inds
        optionally scales y and set bounds
        assumes y in second column
    """
    from sklearn import preprocessing
    with open(yfile) as f:
        y = [x.strip('\n') for x in f.readlines()]
        y = [x.split(' ', 1)[1] for x in y]
        if (header):
            y = y[1:]
        y = np.asarray(y).astype(np.float32)
        if (ind_list is not None):
            y = y[ind_list]
    # optionally scales and set bounds
    if (std):
        y = preprocessing.scale(y)
        if (maxval is not None):
            y[y > maxval] = maxval
            y[y < -maxval] = -maxval
            y = preprocessing.scale(y, with_std=False)
    return y


def filter_k_snps(k=10000):
    (X_train, _, _) = read_gen_k(fdir + 'genosTRN.bin', k=k)
    (X_test, _, _) = read_gen_k(fdir + 'genosTST.bin', k=k)
    y_train = read_yfile(fdir + 'Y_TRN_height.txt', std=False)
    y_test = read_yfile(fdir + 'Y_TST_height.txt', std=False)
    return X_train, X_test, y_train, y_test


def filter_data(pval=0.001):
    (X_train, n_train, p) = read_gen_pval(fdir + 'genosTRN.bin', pval=pval)
    (X_test, n_test, p) = read_gen_pval(fdir + 'genosTST.bin', pval=pval)
    y_train = read_yfile(fdir + 'Y_TRN_height.txt', std=False)
    y_test = read_yfile(fdir + 'Y_TST_height.txt', std=False)
    return X_train, X_test, y_train, y_test


def load_data(small=True, force_pkl_recreation=False, snp_list=None):
    if small:
        if os.path.exists(wdir + dataset_file_sm) and not force_pkl_recreation:
            with open(wdir + dataset_file_sm, "rb") as f:
                X_train, X_test, y_train, y_test = pickle.load(f)
                return X_train, X_test, y_train, y_test

        print("No binary .pkl file has been found for this dataset. The data will "
              "be parsed to produce one. This will take a few minutes.")

        (X_train, n_train, p) = read_gbinfile(cldir + 'genosTRN.mg.bin', std=False)
        (X_test, n_test, p) = read_gbinfile(cldir + 'genosTST.mg.bin', std=False)
        y_train = read_yfile(cldir + 'Y_TRN.mg.txt', std=False)
        y_test = read_yfile(cldir + 'Y_TST.mg.txt', std=False)
        with open(wdir + dataset_file_sm, "wb") as f:
            pickle.dump((X_train, X_test, y_train, y_test), f, pickle.HIGHEST_PROTOCOL)
        return X_train, X_test, y_train, y_test
    else:
        if snp_list == None:
            with open(fdir + 'mengying.snp.list') as f:
                snp_list = [int(x.strip('\n')) for x in f.readlines()]

        if os.path.exists(wdir + dataset_file) and not force_pkl_recreation:
            with open(wdir + dataset_file, "rb") as f:
                X_train, X_test, y_train, y_test = pickle.load(f)
                return X_train, X_test, y_train, y_test

        print("No binary .pkl file has been found for this dataset. The data will "
              "be parsed to produce one. This will take a few minutes.")
        ntrain = 20000
        ntest = 5000
        ind_train = np.linspace(0, 79000, ntrain, dtype=int)
        (X_train, n_train, p) = read_gbinfile(fdir + 'genosTRN.bin', snp_list=snp_list, std=False)
        h_train = read_yfile(fdir + 'Y_TRN_height.txt', std=False)

        ind_test = np.linspace(0, 20000, ntest, dtype=int)
        (X_test, n_test, p) = read_gbinfile(fdir + 'genosTST.bin', snp_list=snp_list, std=False)
        h_test = read_yfile(fdir + 'Y_TST_height.txt', std=False)
        with open(wdir + dataset_file, "wb") as f:
            pickle.dump((X_train, X_test, h_train, h_test), f, pickle.HIGHEST_PROTOCOL)
        return X_train, X_test, h_train, h_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data(force_pkl_recreation=False)
    print(X_train.shape)
    print(X_test)

