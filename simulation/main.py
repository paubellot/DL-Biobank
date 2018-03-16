import os
import numpy as np
import Utils
from models import Train
fdir = '/home/data/biobank/'

def data_rand(k=50000):
    snp_list = np.random.permutation(500000)[0:k]
    x_train, _, _ = Utils.readgbinfile(fdir + 'genosTRN.bin', snp_list=snp_list, std=False)
    x_test, _, _ = Utils.readgbinfile(fdir + 'genosTST.bin', snp_list=snp_list, std=False)
    return x_train, x_test, snp_list


def syn_data_lin(x_train, x_test, shape, scale, h=0.45):
    n_p, n_f = x_train.shape
    s = np.random.gamma(shape, scale, n_f)
    w = s * np.random.permutation(np.asanyarray([-1, 1]*(n_f/2)))
    w = w/sum(w)
    ns = np.sqrt((1-h)/h*np.dot(x_train, w).std()**2)
    y_tr = np.dot(x_train, w) + ns*np.random.randn(n_p)
    ns = np.sqrt((1-h)/h*np.dot(x_test, w).std()**2)
    y_tst = np.dot(x_test, w)+ns*np.random.randn(x_test.shape[0])
    return y_tr, y_tst, w


def syn_data_epis(x_train, x_test,shape,scale,h=0.45):
    n_p, n_f = x_train.shape
    ind = np.random.permutation(n_f)
    ind = ind.reshape(n_f / 2, 2)
    y = np.zeros((x_train.shape[0], 1)).ravel()
    t = np.zeros((x_test.shape[0], 1)).ravel()
    for i in range(0, ind.shape[0]):
        aux = np.prod(x_train[:, ind[i, :]], axis=1)
        E = np.random.gamma(shape, scale, 1) * np.random.permutation(np.asanyarray([-1, 1]))[0]
        y[np.where(aux != 0)[0]] += E
        aux = np.prod(x_test[:, ind[i, :]], axis=1)
        t[np.where(aux != 0)[0]] += E
    p = np.sqrt(np.var(y) * (1. - h) / h)
    ytr = y + p * np.random.randn(y.shape[0], 1).ravel()
    h2 = np.var(y) / np.var(ytr)
    print h2
    ytst = t + p * np.random.randn(t.shape[0], 1).ravel()
    return ytr, ytst


def readgbinfile(binfile, snp_list=None, ind_list=None, std=True, maxval=None):
#------------------------------------------------------------------------------
    """ reads genotype bin file into np array
        Optionally, reads only snps and samples specified in snp_list and ind_list
        by default (std=True) scales X
        optionally (maxval=MaxValue) set bounds
    """
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


def computeCor(y_tr):
    from tqdm import tqdm
    from scipy.stats import linregress
    import struct
    nbyte = 1
    f = open(fdir + 'genosTRN.bin', 'rb')
    n = int(struct.unpack('1d', f.read(8))[0])
    p = int(struct.unpack('1d', f.read(8))[0])
    result_all = []
    pbar = tqdm(total=p)
    for i in range(p):
        # replace 'c' by 'd' if double precision
        line = f.read(nbyte * n)
        temp = np.array(struct.unpack(str(n) + 'b', line)).astype(np.float32)
        slope, _, _, p_value, _ = linregress(temp, y_tr)
        result_all.append([slope, p_value])
        pbar.update(1)
    pbar.close()
    return np.asarray(result_all)


def readGWAS(C,k):
    C[C != C] = 0
    snp_list = C[:, 1].ravel().argsort()
    snp_list = snp_list[0:k]
    x_train, _, _ = Utils.readgbinfile(fdir + 'genosTRN.bin', snp_list=snp_list, std=False)
    x_test, _, _ = Utils.readgbinfile(fdir + 'genosTST.bin', snp_list=snp_list, std=False)
    return x_train, x_test


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="""Simulated data """)
    parser.add_argument('--method',
                        type=str,
                        default="mlp1",
                        help='Method: mlp1, mlp2, lasso or ridge')

    parser.add_argument('--k',
                        type=int,
                        default=10000,
                        help='number of snps: default 10000')

    parser.add_argument('--loci',
                        type=int,
                        default=100,
                        help='number of causal loci snps: default 100')

    parser.add_argument('--genmodel',
                        type=str,
                        default="linear",
                        help='Phenotype model: linear or epistasia. Default linear')

    parser.add_argument('--recreation',
                        type=str2bool,
                        default=False, nargs='?',
                        const=True,
                        help='force recreation (bool): default False')

    parser.add_argument('--shape',
                        type=float,
                        default=5,
                        help='np.random.gamma shape: 5')

    parser.add_argument('--scale',
                        type=float,
                        default=0.25,
                        help='np.random.gamma scale: 0.25')

    parser.add_argument('--h2',
                        type=float,
                        default=0.45,
                        help='heritability: 0.45')
    
    parser.add_argument('--id',
                        type=str,
                        default="0",
                        help='id replica')
    args = parser.parse_args()
    if args.genmodel.lower() == "linear":
        ytr_name = "data/y_train_linear_"+str(args.loci)+"_id"+args.id+".txt"
        ytst_name = "data/y_test_linear_"+str(args.loci)+"_id"+args.id+".txt"
        snp_name = "data/snplist_linear_"+str(args.loci)+"_id"+args.id+".txt"
    elif args.genmodel.lower() == "epistasia":
        ytr_name = "data/y_train_epis_" + str(args.loci) +"_id"+args.id+ ".txt"
        ytst_name = "data/y_test_epis_" + str(args.loci) +"_id"+args.id+ ".txt"
        snp_name = "data/snplist_epis_" + str(args.loci) +"_id"+args.id+ ".txt"
    else:
        raise argparse.ArgumentTypeError('genmodel: linear or epistasia are only supported.')
    C_name = "data/Corr_"+str(args.loci)+args.genmodel+"_id"+args.id+".csv"
    if os.path.exists(ytr_name) and not args.recreation:
        print("Reading phenotype data: " + ytr_name)
        y_tr = np.loadtxt(ytr_name, delimiter=",")
        y_tst = np.loadtxt(ytst_name, delimiter=",")
    else:
        print("Generating phenotype")
        x_train, x_test, snp_list = data_rand(k=args.loci)
        if args.genmodel.lower() == "linear":
            print("Linear model")
            y_tr, y_tst, w = syn_data_lin(x_train, x_test, shape=args.shape, scale=args.scale)
        elif args.genmodel.lower() == "epistasia":
            print("Epistasia model")
            y_tr, y_tst = syn_data_epis(x_train, x_test, shape=args.shape, scale=args.scale)
        else:
            raise argparse.ArgumentTypeError('Linear or epistasia are only supported.')
        print("Saving to: "+ ytr_name)
        np.savetxt(snp_name,snp_list,delimiter=",")
        np.savetxt(ytr_name, y_tr, delimiter=',')
        np.savetxt(ytst_name, y_tst, delimiter=',')
    if os.path.exists(C_name) and not args.recreation:
        C = np.loadtxt(C_name, delimiter=",")
    else:
        print("Computing GWAS")
        C = computeCor(y_tr)
        print("Saving to :" + C_name)
        np.savetxt(C_name, C, delimiter=",")

    xTr, xTst = readGWAS(C, args.k)
    r = Train(xTr, xTst, y_tr, y_tst, args.method)
    print([args,"r2 = "+str(r)])
    fname = "results/out.txt"
    with open(fname, 'a') as fh:
        fh.write([args,"r2 = "+str(r)+'\n'])