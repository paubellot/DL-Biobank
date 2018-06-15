from .hck_data import DataDefinition
import pandas as pd
import numpy as np
import os
import hickle as hkl


def clean_data(trait, k, tr =True):
    Ds = DataDefinition(trait, k)
    if tr:
        markers = Ds.markers_tr()
        pheno = Ds.pheno_tr()
    else:
        markers = Ds.markers_tst()
        pheno = Ds.pheno_tst()
    has_trait_data = pd.DataFrame(pheno).notnull().values.ravel()
    return markers[has_trait_data,:], pheno[has_trait_data]


def retrieve_data(trait, k, unif=False):
    Ds = DataDefinition(trait_name=trait, k=k, unif=unif)
    if os.path.exists(Ds._hdf5_file):
        aux = hkl.load(Ds._hdf5_file)
        Ds._markers_tr = aux["x_tr"]
        Ds._markers_tst = aux["x_tst"]
        Ds._pheno_tr = aux["y_tr"]
        Ds._pheno_tst = aux["y_tst"]
    else:
        Ds.saveHDF5()

    xtr = Ds._markers_tr
    ytr = Ds._pheno_tr
    has_trait_data = pd.DataFrame(ytr).notnull().values.ravel()
    xtr, ytr = xtr[has_trait_data, :], ytr[has_trait_data]
    xtst = Ds._markers_tst
    ytst = Ds._pheno_tst
    has_trait_data = pd.DataFrame(ytst).notnull().values.ravel()
    xtst, ytst = xtst[has_trait_data, :], ytst[has_trait_data]
    return (xtr, xtst, ytr, ytst)


def convert_to_individual_alleles(array):
    """
    Convert SNPs to individual copies so neuralnet can learn dominance relationships.
    [-1, 0, 1] => [(0, 0), (0, 1), (1, 1)] => [0, 0, 0, 1, 1, 1]
    """
    # Set non-integer values to 0 (het)
    array = np.trunc(array)
    incr = array  # Now we have 0, 1, and 2
    incr = incr[:,:,np.newaxis] # Add another dimension.
    pairs = np.pad(incr, ((0,0), (0,0), (0,1)), mode='constant') # Append one extra 0 value to final axis.
    twos = np.sum(pairs, axis=2) == 2
    pairs[twos] = [1,1]
    x, y, z = pairs.shape
    pairs = pairs.reshape((x, y*z)) # Merge pairs to one axis.
    return pairs