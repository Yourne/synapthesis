import pandas as pd
from os import path
import os
from sklearn.svm import OneClassSVM
from scipy.stats import boxcox
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import time


MODEL = "oc-svm"
DATADIR = "data"
OUTDIR = "dataout"
AWARD_PROCEDURE = "aperta"


def outlier_proportion_score(preds, n_outliers):
    # estimated correct proportion of outliers
    c = n_outliers / len(preds)
    # number of predicted outliers
    x = (sum(preds) - len(preds)) / 2
    return -x**2 - c


if __name__ == "__main__":
    fname = AWARD_PROCEDURE + ".csv"
    dataset = pd.read_csv(path.join(DATADIR, fname), index_col="idx")

    # remove oggetto, id_be, id_pa, id_lsf, data_inizio, data_fine
    X = dataset.drop(columns=[
        "oggetto", "id_lotto", "id_pa", "id_be", "id_lsf", "data_inizio",
        "data_fine"])
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

    # optimize the hyperparameters
    n_outliers = 19
    params = {"nu": [n_outliers / X.shape[0]]}

    # GridSearchCV requires a scoring function.
    # The proportion of outliers, assuming to know it.
    score = make_scorer(outlier_proportion_score)
    grid = GridSearchCV(OneClassSVM(), params, verbose=1, cv=3, scoring=score)
    start = time.time()
    grid.fit(X)
    elapsed = time.time() - start

    print(f"dataset {fname[:-4]}")
    print(f"hyperparameter optimization required {elapsed:.2f} s")
    print("best nu: {0}".format(grid.best_estimator_.nu))

    # use the best estimator to compute the kernel density estimate
    model = grid.best_estimator_

    # append score to dataset
    dataset["score"] = model.score_samples(X)
    dataset["score"] = model.predict(X)
    dataset = dataset.sort_values(by="score")

    # select saving position
    outpath = path.join(OUTDIR, AWARD_PROCEDURE, MODEL)

    # create folders if needed
    try:
        os.makedirs(outpath)
    except FileExistsError:
        pass

    # create name
    fname = AWARD_PROCEDURE + "_" + MODEL + "_score.csv"
    fname = os.path.join(outpath, fname)

    # save the files for Davide
    # dataset.iloc[:100].to_csv(fname)

    # save files for the outliers analyzer.py
    dataset["score"].to_csv(fname)
