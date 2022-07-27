import pandas as pd
from os import path
import numpy as np
import matplotlib.pyplot as plt


def output_flagged_contracts(X, preds, model):
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
    output_path = path.join(output_directory, model + ".csv")

    outliers = lotti.loc[X.iloc[idx].index]
    outliers.to_csv(output_path)


def plot(X, preds, model, show=False):
    _, ax = plt.subplots(1, 3, figsize=(6.4*3, 4.2), sharey=True)
    for i, feature in enumerate(["median_annual_revenue",
                                "median_annual_expenditure", "duration"]):
        s = ax[i].scatter(x=X[feature], y=X.sum_total, c=preds, alpha=1, s=.2)
        ax[i].legend(*s.legend_elements())
        ax[i].set_ylabel("contract sum total")
        ax[i].set_xlabel(feature)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
    plt.tight_layout()
    fname = path.join("output", model)
    plt.savefig(fname + ".png")
    if show:
        plt.show()