import pandas as pd
import numpy as np
from os import path
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


def get_contract_id_lotto(X, preds):
    """preds: output of a sklearn model"""
    idx = np.argwhere(preds == -1).squeeze()
    return X.iloc[idx].index


def plot(X, preds):
    _, ax = plt.subplots(1, 3, figsize=(6.4*3, 4.2), sharey=True)
    for i, feature in enumerate(["median_annual_revenue",
                                "median_annual_expenditure", "duration"]):
        ax[i].scatter(x=X[feature], y=X.value, c=preds, alpha=.3, s=.2)
        ax[i].legend()
        ax[i].set_ylabel("contract value")
        ax[i].set_xlabel(feature)
        if feature == "duration":
            ax[i].set_yscale("log")
        else:
            ax[i].loglog()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    directory = "datasets"
    file_name = "appMed_adesione"
    X = pd.read_csv(
        path.join(directory, file_name), index_col="id_lotto")
    clf = IsolationForest()
    preds = clf.fit_predict(X)
    plot(X, preds)
