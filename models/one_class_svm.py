import pandas as pd
from os import path
from sklearn.svm import OneClassSVM
from scipy.stats import boxcox
from sklearn.preprocessing import RobustScaler
from utils import plot, output_flagged_contracts


if __name__ == "__main__":
    directory = "datasets"
    file_name = "appMed_aperta"
    model = "oc_svm"
    X = pd.read_csv(
        path.join(directory, file_name + ".csv"), index_col="id_lotto")

    # remove time related features
    X = X.drop(columns=["sinDayOfYear", "cosDayOfYear", "daysSinceBaseDate"])
    # remove median contract pa and median contract be as they are computed
    # on whole dataset, not only the CPV and award procedure
    X = X.drop(columns=["median_contract_pa", "median_contract_be"])

    # preprocessing
    col_names = X.columns
    X.duration = X.duration.replace(0, X.duration.median())
    scaler = RobustScaler(with_centering=False)
    table = scaler.fit_transform(X)
    # scale only the real-valued columns
    for i in range(0, 2):
        table[:, i], _ = boxcox(table[:, i])

    # classify
    clf = OneClassSVM(nu=.001, verbose=1)
    preds = clf.fit_predict(table)
    print(f"total flagged outliers: {sum(preds == -1)}")
    plot(X, preds, model, file_name)
    # append anomaly score to df
    s = clf.score_samples(table)
    output_flagged_contracts(X, preds, model, file_name)
