from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from os import path

if __name__ == "__main__":
    directory = "datasets"
    file_name = "appMed_adesione"
    X = pd.read_csv(
        path.join(directory, file_name), index_col="id_lotto")

    # preprocessing
    duration = X["duration"]
    X = X.drop(columns=["duration"])
    X = np.log10(X+1)
    X["duration"] = duration

    # # use grid search cross-validation to optimize the bandwidth
    # params = {"bandwidth": np.logspace(-1, 1, 20)}
    # print(params)
    # grid = GridSearchCV(KernelDensity(), params, verbose=1)
    # grid.fit(X.values)

    # print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

    # # use the best estimator to compute the kernel density estimate
    # kde = grid.best_estimator_

    kde = KernelDensity(bandwidth=1, kernel="gaussian")
    kde.fit(X.values)

    loglike = kde.score_samples(X.values)
    for thr in np.arange(.1e-7, 1.1e-7, .1e-7):
        # percentage of samples within level threshold
        perc = np.mean(thr < np.exp(loglike))
        total = np.sum(thr > np.exp(loglike))
        print(f"total flagged outliers: {total}")
