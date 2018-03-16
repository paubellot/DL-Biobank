from ML import *
import Utils
import numpy as np
import pandas as pd

gn_dir = "data/genotypes/"
ph_dir = "data/phenotypes/"

def main(trait="all",th=65.975):
    assert bool(set(["all", "BHD", "BMI", "DBP", "Height", "Neuroticism", "Pulse",
                     "SBP", "Weight", "WHR"]) & set([trait])), "%r is not a valid trait" % trait
    if trait == "all":
        trait = ["BHD", "BMI", "DBP", "Height", "Neuroticism", "Pulse", "SBP", "Weight", "WHR"]

    mat = np.asarray(['', 'MLP1', 'MLP2', "Lasso", "Ridge"])
    for t in trait:
        mat = np.vstack((mat,np.array([t, 0, 0, 0, 0])))

    metrics = pd.DataFrame(data=mat[1:, 1:], index=mat[1:, 0], columns=mat[0, 1:])
    for idx, t in enumerate(trait):
        print("Analyzing trait " + t)
        x_train, x_test, y_train, y_test = Utils.load_data(ph_dir, gn_dir, trait=t, th=th)
        metrics.iloc[idx] = np.asarray(Train(x_train, y_train, x_test, y_test))

    print metrics

if __name__ == "__main__":
        main(trait="Height", th=65.975)