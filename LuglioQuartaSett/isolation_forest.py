import pandas as pd
import numpy as np
from os import path
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def output_flagged_contracts(X, preds):
    """preds: output of a sklearn model"""
    # read original dataset
    import_directory = "synData6July"
    lotti_fn = "export_lotti_veneto_2016_2018_giulio_v2.csv"
    lotti_path = path.join(import_directory, lotti_fn)
    lotti = pd.read_csv(lotti_path, index_col="id_lotto")

    # get the indices of the predictions
    idx = np.argwhere(preds == -1).squeeze()
    # output the flagged contracts in csv file
    output_directory = "output"
    output_path = path.join(output_directory, "isolation_forest.csv")
    lotti.loc[X.iloc[idx].index].to_csv(output_path)


def plot(X, preds):
    _, ax = plt.subplots(1, 3, figsize=(6.4*3, 4.2), sharey=True)
    for i, feature in enumerate(["median_annual_revenue",
                                "median_annual_expenditure", "duration"]):
        s = ax[i].scatter(x=X[feature], y=X.value, c=preds, alpha=.3, s=.2)
        ax[i].legend(*s.legend_elements())
        ax[i].set_ylabel("contract value")
        ax[i].set_xlabel(feature)
        if feature == "duration":
            ax[i].set_yscale("log")
        else:
            ax[i].loglog()
    plt.tight_layout()
    plt.show()


def pca_scatter(X, preds):
    pca = PCA(n_components=3)
    Xt = pca.fit_transform(X)
    # print(pca.explained_variance_ratio_)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], c=preds, s=.4)
    plt.show()


if __name__ == "__main__":
    directory = "datasets"
    file_name = "appMed_adesione"
    X = pd.read_csv(
        path.join(directory, file_name), index_col="id_lotto")
    clf = IsolationForest(contamination=.001)
    preds = clf.fit_predict(X.values)
    # i risultati sono sbagliati.
    print(sum(preds == -1))
    plot(X, preds)
    # output_flagged_contracts(X, preds)
