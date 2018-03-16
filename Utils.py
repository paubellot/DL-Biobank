import pandas as pd
import numpy as np
import sys
from sklearn import preprocessing
import struct
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import pickle


""" 
 This function reads genotype file, if present, snp_list specifies read snps, and ind_list individual list
 reads genotype bin file into np array
        Optionally, reads only snps and samples specified in snp_list and ind_list
        by default (std=True) scales X
        optionally (maxval=MaxValue) set bounds
"""
def readgbinfile(binfile, snp_list=None, ind_list=None, std=True, maxval=None):
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
        use_ind = np.ones(n, dtype=bool)
    else:
        use_ind = np.zeros(n, dtype=bool)
        use_ind[ind_list] = True
    # reads all genotypes for ith snp, and append only those needed
    X = []
    for i in range(p):
        # replace 'c' by 'd' if double precision
        line = f.read(nbyte*n)
        if use_snp[i]:
            temp = np.array(struct.unpack(str(n)+'b', line)).astype(np.float32)
            X.append(temp[use_ind])
    # X should be n x p
    X = np.array(X).T
    # optionally scales and set bounds
    if std:
        X = preprocessing.scale(X)
        if maxval is not None:
            X[X > maxval] = maxval
            X[X < -maxval] = -maxval
            X = preprocessing.scale(X, with_std=False)
    f.close()
    return X


def load_data(ph_dir, gn_dir, trait="height", th=65.975):
    assert bool(set(["BHD", "BMI", "DBP", "Height", "Neuroticism", "Pulse",
                     "SBP", "Weight", "WHR"]) & set([trait])), "%r is not a valid trait" % trait
    if trait == "BHD":
        pvalue_file = ph_dir + "pvalue_height.txt"
        pvalue_id = "P-GWAS"
        trait_id = "adjusted_height"
    if trait == "BMI":
        pvalue_file = ph_dir + "pvalue_BMI.txt"
        pvalue_id = "P-GWAS"
        trait_id ="adjusted_BMI"
    if trait == "DBP":
        pvalue_file = ph_dir + "DBP_pvalues.txt"
        pvalue_id = "p-value"
        trait_id = "adjusted_DBP"
    if trait == "Height":
        pvalue_file = ph_dir + "pvalue_height.txt"
        pvalue_id = "P-GWAS"
        trait_id = "adjusted_height"
    if trait == "Neuroticism":
        pvalue_file = ph_dir + "neuroticism_pvalues.txt"
        pvalue_id = "p-value"
        trait_id = "adjusted_log_neuroticism"
    if trait == "Pulse":
        pvalue_file = ph_dir + "pulse_pvalues.txt"
        pvalue_id = "p-value"
        trait_id = "adjusted_pulse"
    if trait == "SBP":
        pvalue_file = ph_dir + "SBP_pvalues.txt"
        pvalue_id = "p-value"
        trait_id = "adjusted_SBP"
    if trait == "Weight":
        pvalue_file = ph_dir + "weight_pvalues.txt"
        pvalue_id = "p-value"
        trait_id = "adjusted_weight"
    if trait == "WHR":
        pvalue_file = ph_dir + "WHR_pvalues.txt"
        pvalue_id = "p-value"
        trait_id = "adjusted_WHR"

    tr_file = ph_dir + trait + "_tr.csv"
    tst_file = ph_dir + trait + "_tst.csv"
    tb = pd.read_csv(pvalue_file, sep=",")
    pval = tb[pvalue_id].values
    pval[pval != pval] = 100
    snp_list = np.where(-10 * np.log(pval) >= th)[0]
    tb = pd.read_csv(tr_file)
    pheno_trn = tb[trait_id].values
    ids_trn = np.where(pheno_trn == pheno_trn)[0]
    x_train = readgbinfile(gn_dir + 'genosTRN.bin', snp_list=snp_list, ind_list=ids_trn, std=False)
    tb = pd.read_csv(tst_file)
    pheno_tst = tb[trait_id].values
    ids_tst = np.where(pheno_tst == pheno_tst)[0]
    x_test = readgbinfile(gn_dir + 'genosTST.bin', snp_list=snp_list, ind_list=ids_tst, std=False)
    return x_train, x_test, pheno_trn[ids_trn], pheno_tst[ids_tst]




