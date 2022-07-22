import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM


def plot(X, preds):
    _, ax = plt.subplots(1, 3, figsize=(6.4*3, 4.2), sharey=True)
    for i, feature in enumerate(["median_annual_revenue",
                                "median_annual_expenditure", "duration"]):
        s = ax[i].scatter(x=X[feature], y=X.value, c=preds, alpha=.3, s=.2)
        ax[i].legend(*s.legend_elements())
        ax[i].set_ylabel("contract value")
        ax[i].set_xlabel(feature)
    plt.tight_layout()
    plt.show()


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

    # classify
    clf = OneClassSVM(nu=.01, verbose=1)
    preds = clf.fit_predict(X.values)
    print(f"total flagged outliers: {sum(preds == -1)}")
    # sono troppi
    plot(X, preds)
