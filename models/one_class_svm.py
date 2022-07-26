import pandas as pd
from os import path
from sklearn.svm import OneClassSVM
from scipy.stats import boxcox
from sklearn.preprocessing import RobustScaler
from utils import plot, output_flagged_contracts


if __name__ == "__main__":
    directory = "datasets"
    file_name = "appMed_adesione"
    X = pd.read_csv(
        path.join(directory, file_name), index_col="id_lotto")

    # preprocessing
    col_names = X.columns
    X.duration = X.duration.replace(0, X.duration.median())
    scaler = RobustScaler(with_centering=False)
    t = scaler.fit_transform(X)
    for i in range(0, 6):
        t[:, i], _ = boxcox(t[:, i])

    # classify
    clf = OneClassSVM(nu=.001, verbose=1)
    preds = clf.fit_predict(t)
    print(f"total flagged outliers: {sum(preds == -1)}")
    plot(X, preds, "one-class_svm")
    # append anomaly score to df
    s = clf.score_samples(t)
    output_flagged_contracts(X, preds, "one-class_svm")
