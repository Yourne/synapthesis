from sklearn.neighbors import KernelDensity
# from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from scipy.stats import boxcox
import pandas as pd
import numpy as np
from os import path

if __name__ == "__main__":
    directory = "datasets"
    file_name = "appMed_adesione"
    X = pd.read_csv(
        path.join(directory, file_name), index_col="id_lotto")

    # preprocessing
    col_names = X.columns
    X.duration = X.duration.replace(0, X.duration.median())
    scaler = RobustScaler(with_centering=False)
    X = scaler.fit_transform(X)
    for i in range(0, 6):
        X[:, i], _ = boxcox(X[:, i])

    # # use grid search cross-validation to optimize the bandwidth
    # params = {"bandwidth": np.logspace(-1, 1, 20)}
    # print(params)
    # grid = GridSearchCV(KernelDensity(), params, verbose=1)
    # grid.fit(X)

    # print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    # # best bandwith: 0.6951927961775606

    # # use the best estimator to compute the kernel density estimate
    # kde = grid.best_estimator_

    bandwidth = 0.26366508987303583
    kde = KernelDensity(bandwidth=1, kernel="gaussian")
    kde.fit(X)

    loglike = kde.score_samples(X)
    for thr in np.linspace(1e-8, 1e-7, 10):
        total = sum(np.exp(loglike) < thr)
        alpha = total / len(X)
        print(f"thr: {thr} outliers {total}, alpha: {alpha}")
