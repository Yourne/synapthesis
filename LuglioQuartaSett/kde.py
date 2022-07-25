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

    # # # use grid search cross-validation to optimize the bandwidth
    # params = {"bandwidth": np.logspace(-1, 1, 20)}
    # print(params)
    # grid = GridSearchCV(KernelDensity(), params, verbose=1)
    # grid.fit(X.values)

    # print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    # best bandwith: 0.6951927961775606

    # use the best estimator to compute the kernel density estimate
    # kde = grid.best_estimator_
    bandwidth = 0.6951927961775606
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(X.values)

    loglike = kde.score_samples(X.values)
    for thr in np.linspace(4e-7, 5e-7, 10):
        # percentage of samples within level threshold
        perc = np.mean(thr < np.exp(loglike))
        total = np.sum(thr > np.exp(loglike))
        print(f"flagged outliers: {total}, th: {thr}")
