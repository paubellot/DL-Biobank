import os
import numpy as np
import pandas as pd
from scipy import stats
import hickle as hkl

_data_dir = '/home/data/biobank/'
_mydata_dir = os.path.join(os.path.expanduser("~"), "Data/bbUK/")

def readgbinfile(binfile, snp_list=None, ind_list=None, std=True, maxval=None):
#------------------------------------------------------------------------------
    """ reads genotype bin file into np array
        Optionally, reads only snps and samples specified in snp_list and ind_list
        by default (std=True) scales X
        optionally (maxval=MaxValue) set bounds
    """
    from sklearn import preprocessing
    import struct
    nbyte = 1
    f = open(binfile, 'rb')
    n = int(struct.unpack('1d', f.read(8))[0])
    p = int(struct.unpack('1d', f.read(8))[0])
    if snp_list is None:
        use_snp = np.ones(p, dtype=bool)
    else:
        use_snp = np.zeros(p, dtype=bool)
        use_snp[snp_list] = True
    if ind_list is None:
        use_ind = np.ones(n,dtype=bool)
    else:
        use_ind = np.zeros(n,dtype=bool)
        use_ind[ind_list] = True
    # reads all genotypes for ith snp, and append only those needed
    X = []
    for i in range(p):
        # replace 'c' by 'd' if double precision
        line = f.read(nbyte*n)
        if use_snp[i]:
            temp = np.array(struct.unpack( str(n)+'b', line)).astype(np.float32)
            X.append(temp[use_ind])
    # X should be n x p
    X = np.array(X).T
    # optionally scales and set bounds
    if std:
        X = preprocessing.scale(X)
        if maxval is not None:
            X[X > maxval]  = maxval
            X[X < -maxval] = -maxval
            X = preprocessing.scale(X,with_std=False)
    f.close()
    return X


class DataDefinition(object):
    def __init__(self, trait_name, k=10000, unif=False):
        """
        Defines a datatype to predict.

        Parameters:

            trait_name: The name of the trait column in phenotypes.csv to predict.

            *args:      The tokens of the path to the data directory that contains
                        phenotypes.csv and genotypes.csv.

            **kwargs:   Can contain compressed_geno=True to load a bz2 compressed
                        genotypes file.
        """

        self.trait_name = trait_name
        self.train_markers_path = os.path.join(_data_dir, "genosTRN.bin")
        self.test_markers_path= os.path.join(_data_dir, "genosTST.bin")
        self.pheno_tr_path = os.path.join(_mydata_dir, "ph_tr.csv")
        self.pheno_tst_path = os.path.join(_mydata_dir, "ph_tst.csv")
        self.pval_path = os.path.join(_mydata_dir, "pvals.csv")
        self._snp_list = None
        self._k = k
        self._pvals = None
        self._markers_tr = None
        self._pheno_tr = None
        self._markers_tst = None
        self._pheno_tst = None
        self._unif = unif
        self._hdf5_file = os.path.join(_mydata_dir, "traits", self.trait_name,
                                      str(self._k) +("_unif" if self._unif else "_best")+".hkl")

    def pheno_tr(self):
        if self._pheno_tr is None:
            if not os.path.exists(self._hdf5_file):
                params = {'index_col': None, 'sep': ','}
                self._pheno_tr = pd.read_csv(self.pheno_tr_path, **params)[self.trait_name].values
            else:
                aux = hkl.load(self._hdf5_file)
                self._pheno_tst = aux["y_tr"]
        return self._pheno_tr

    def pheno_tst(self):
        if self._pheno_tst is None:
            if not os.path.exists(self._hdf5_file):
                params = {'index_col': None, 'sep': ','}
                self._pheno_tst = pd.read_csv(self.pheno_tst_path, **params)[self.trait_name].values
            else:
                aux = hkl.load(self._hdf5_file)
                self._pheno_tst = aux["y_tst"]
        return self._pheno_tst

    def markers_tr(self):
        if self._markers_tr is None:
            if not os.path.exists(self._hdf5_file):
                if self._snp_list is None:
                    if self._pvals is None:
                        params = {'index_col': None, 'sep': ','}
                        self._pvals = pd.read_csv(self.pval_path, **params)[self.trait_name].values
                    self._pvals[self._pvals != self._pvals] = 100
                    snp_list = self._pvals.ravel().argsort()
                    snp_list = snp_list[0:self._k]
                    snp_list = np.sort(snp_list)
                    self._snp_list = snp_list
                self._markers_tr = readgbinfile(self.train_markers_path, snp_list=self._snp_list, std=False)
            else:
                aux = hkl.load(self._hdf5_file)
                self._markers_tr = aux["x_tr"]
        return self._markers_tr

    def markers_tst(self):
        if self._markers_tst is None:
            if not os.path.exists(self._hdf5_file):
                if self._snp_list is None:
                    if self._pvals is None:
                        self._pvals = pd.read_csv(self.pval_path)[self.trait_name].values
                    self._pvals[self._pvals != self._pvals] = 100
                    snp_list = self._pvals.ravel().argsort()
                    snp_list = snp_list[0:self._k]
                    snp_list = np.sort(snp_list)
                    self._snp_list = snp_list
                self._markers_tst = readgbinfile(self.test_markers_path, snp_list=self._snp_list, std=False)
            else:
                aux = hkl.load(self._hdf5_file)
                self._markers_tr = aux["x_tst"]
        return self._markers_tst

    def markers_cnn_tr(self):
        if self._markers_tr is None:
            if not os.path.exists(self._hdf5_file):
                if self._markers_tst is None:
                    if self._pvals is None:
                        self._pvals = pd.read_csv(self.pval_path)[self.trait_name].values
                    self._pvals[self._pvals != self._pvals] = 100
                    df = pd.read_csv(os.path.join(_mydata_dir, "ukb_snpinfo.csv"))
                    chr_info = pd.read_csv(os.path.join(_mydata_dir, "chr_info.csv"))
                    length = chr_info["max"].values - chr_info["min"].values
                    chromosomes = range(1, 23)
                    chromosomes.append(25)
                    L = sum(length[np.array(chromosomes) - 1])
                    lc = length / (L / self._k)
                    snp_list = []
                    for chr in chromosomes:
                        positions = df.loc[df['chromosome'] == chr]['position'].values
                        if len(positions) > lc[chr - 1]:
                            chunks = np.array_split(positions, lc[chr - 1])
                            for c in chunks:
                                s = stats.describe(c)[1]
                                i = (df['chromosome'] == chr) & (df['position'] >= s[0]) & (df['position'] <= s[1])
                                pos = np.where(i.values)[0]
                                snp_list.append(pos[self._pvals[pos].argsort()][0])
                    snp_list = np.asanyarray(snp_list).ravel()
                    snp_list = np.sort(snp_list)
                    self._snp_list = snp_list
                self._markers_tr = readgbinfile(self.train_markers_path, snp_list=self._snp_list, std=False)
            else:
                aux = hkl.load(self._hdf5_file)
                self._markers_tr = aux["x_tr"]
        return self._markers_tr

    def markers_cnn_tst(self):
        if self._markers_tst is None:
            if not os.path.exists(self._hdf5_file):
                if self._snp_list is None:
                    if self._pvals is None:
                        self._pvals = pd.read_csv(self.pval_path)[self.trait_name].values
                    self._pvals[self._pvals != self._pvals] = 100
                    df = pd.read_csv(os.path.join(_mydata_dir, "ukb_snpinfo.csv"))
                    chr_info = pd.read_csv(os.path.join(_mydata_dir, "chr_info.csv"))
                    length = chr_info["max"].values - chr_info["min"].values
                    chromosomes = range(1, 23)
                    chromosomes.append(25)
                    L = sum(length[np.array(chromosomes) - 1])
                    lc = length / (L / self._k)
                    snp_list = []
                    for chr in chromosomes:
                        positions = df.loc[df['chromosome'] == chr]['position'].values
                        if len(positions) > lc[chr - 1]:
                            chunks = np.array_split(positions, lc[chr - 1])
                            for c in chunks:
                                s = stats.describe(c)[1]
                                i = (df['chromosome'] == chr) & (df['position'] >= s[0]) & (df['position'] <= s[1])
                                pos = np.where(i.values)[0]
                                snp_list.append(pos[self._pvals[pos].argsort()][0])
                    snp_list = np.asanyarray(snp_list).ravel()
                    snp_list = np.sort(snp_list)
                    self._snp_list = snp_list
                self._markers_tst = readgbinfile(self.test_markers_path, snp_list=self._snp_list, std=False)
            else:
                aux = hkl.load(self._hdf5_file)
                self._markers_tr = aux["x_tst"]
        return self._markers_tst

    def saveHDF5(self):
        if self._unif:
            x_tr = self.markers_cnn_tr()
            x_tst = self.markers_cnn_tst()
        else:
            x_tr = self.markers_tr()
            x_tst = self.markers_tst()
        y_tr = self.pheno_tr()
        y_tst = self.pheno_tst()
        s = {"x_tr": x_tr, "x_tst": x_tst, "y_tr": y_tr, "y_tst": y_tst}
        hkl.dump(s, self._hdf5_file, mode='w', compression='gzip')
