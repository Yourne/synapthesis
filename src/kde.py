from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from scipy.stats import boxcox
import pandas as pd
import numpy as np
import os
import time


DATADIR = "data"
OUTDIR = "dataout"
MODEL = "kde"


def strjoin(*words, separator="_"):
    s = words[0]
    for w in words[1:]:
        s += separator + w
    return s


if __name__ == "__main__":
    award_procedure = "aperta"
    split = "train"
    fname = award_procedure + "_" + split + ".csv"
    dataset = pd.read_csv(os.path.join(DATADIR, fname), index_col="idx")

    # remove oggetto, id_be, id_pa, id_lsf, data_inizio, data_fine
    X = dataset.drop(columns=[
        "oggetto", "id_lotto", "id_pa", "id_be", "id_lsf", "data_inizio",
        "data_fine", "outlier"])
    # remove time related features
    X = X.drop(columns=["daysSinceBaseDate", 'sinMonth', 'cosMonth'])
    # remove median contract pa, median contract be, "pa_med_ann_n_contr",
    # "be_med_ann_n_contr" as they are computed on whole dataset, not only
    # the CPV and award procedure
    X = X.drop(columns=['pa_med_ann_contr', 'be_med_ann_contr',
                        'pa_med_ann_n_contr', 'be_med_ann_n_contr'])

    # print(X.columns)
    X = X.drop(columns=[
        'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7',
        'month_8', 'month_9', 'month_10', 'month_11', 'month_12'
    ])
    # 'amount', 'pa_med_ann_expenditure', 'be_med_ann_revenue', 'duration'

    # preprocessing
    X.duration = X.duration.replace(0, X.duration.median())
    scaler = RobustScaler(with_centering=False)
    X = scaler.fit_transform(X)
    # scale only the real-valued columns
    for i in range(0, 3):
        X[:, i], _ = boxcox(X[:, i])

    # optimize the bandwidth
    params = {"bandwidth": np.logspace(-2, 1, 4)}

    grid = GridSearchCV(KernelDensity(), params, verbose=1, cv=3)
    start = time.time()
    grid.fit(X)
    elapsed = time.time() - start

    print(f"dataset {fname[:-4]}")
    print(f"bandwith hyperparameter search required {elapsed:.2f} s")
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

    # use the best estimator to compute the kernel density estimate
    kde = grid.best_estimator_

    # likelihoods
    dataset["score"] = kde.score_samples(X)
    dataset = dataset.sort_values(by="score")

    # select saving position
    outpath = os.path.join(OUTDIR, award_procedure, MODEL)

    # create folders if needed
    try:
        os.makedirs(outpath)
    except FileExistsError:
        pass

    # create filename
    fname = strjoin(award_procedure, MODEL, split, "score.csv")
    fname = os.path.join(outpath, fname)

    # save the files for Davide
    # dataset.iloc[:100].to_csv(fname)

    # save files for the outliers analyzer.py
    dataset["score"].to_csv(fname)
