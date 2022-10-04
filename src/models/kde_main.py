from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from scipy.stats import boxcox
import numpy as np
import time

MODEL = "kde"


def preprocess(dataset):
    # drop cols
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
    # remainng features:
    # 'amount', 'pa_med_ann_expenditure', 'be_med_ann_revenue', 'duration'
    # preprocessing
    X.duration = X.duration.replace(0, X.duration.median())
    scaler = RobustScaler(with_centering=False)
    X = scaler.fit_transform(X)
    # scale only the real-valued columns
    for i in range(0, 3):
        X[:, i], _ = boxcox(X[:, i])
    return X


def train(X):
    # train and optimize hyperparameters
    params = {"bandwidth": np.logspace(-2, 1, 4)}

    grid = GridSearchCV(KernelDensity(), params, verbose=1, cv=3)
    start = time.time()
    grid.fit(X)
    elapsed = time.time() - start

    print(f"bandwith hyperparameter search required {elapsed:.2f} s")
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

    return grid.best_estimator_


def test(kde, X_te):
    return kde.score_samples(X_te)
