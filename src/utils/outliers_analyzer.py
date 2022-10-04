from os import path
import os
import pandas as pd
import utils.prep as prep
import matplotlib.pyplot as plt
import numpy as np
from utils import strjoin

OUTDIR = "dataout"
AWARD_PROCEDURE = "aperta"
CPV = None
MODEL = "oc-svm"
SPLIT = "train"
N_OUTLIERS = 5

# USAGE DESIGN:
# specify award procedure, cpv (if-any), and the model to eveluate


def plot_outlier_distr(N_OUTLIERS, resultsExt, df, outpath):
    # plot be and pa local and global amount distributions
    resultsExt = resultsExt.sort_values(by="score")
    for i in range(N_OUTLIERS):
        # get variables
        pa = int(resultsExt.iloc[i].id_pa)
        be = int(resultsExt.iloc[i].id_be)
        score = np.round(resultsExt.iloc[i].score, 4)
        lot_id = str(int(resultsExt.iloc[i].id_lotto))
        lot_amount = resultsExt.iloc[i].amount

        outlier_outpath = path.join(outpath, lot_id)
        try:
            os.makedirs(outlier_outpath)
        except FileExistsError:
            pass

        for scope, ds in {"local": resultsExt, "global": df}.items():
            for agent, agent_id in {"pa": pa, "be": be}.items():
                log_amount = np.log10(ds[ds["id_" + agent] == agent_id].amount)
                fig, ax = plt.subplots()
                ax.set_xscale("log")
                ax.hist(log_amount)
                ax.vlines(np.log10(lot_amount), 0, 1,
                          transform=ax.get_xaxis_transform(), colors="r")
                ax.set_title(agent + ": " + str(agent_id) + " score: " +
                             str(score))
                ax.set_xlabel("lot value")
                ax.set_ylabel("lot count")
                figure_name = agent + "-" + scope + "-" + "amount_distribution"
                fname = path.join(outlier_outpath, figure_name + ".png")
                plt.savefig(fname)
                plt.close(fig)


def scatterscore(X, outpath, features):
    n = len(features)
    fig, ax = plt.subplots(1, n, figsize=(6.4 * n, 4.2), sharey=True)
    if CPV is not None:
        fig.suptitle(f"{MODEL} {AWARD_PROCEDURE} {CPV}")
    else:
        fig.suptitle(f"{MODEL} {AWARD_PROCEDURE}")
    for i, feature in enumerate(features):
        s = ax[i].scatter(x=X[feature], y=X.amount, c=X.score, alpha=1, s=.2)
        ax[i].legend(*s.legend_elements())
        ax[i].set_ylabel("lot amount")
        ax[i].set_xlabel(feature)
        # ax[i].set_xscale("log")
        # ax[i].set_yscale("log")
    plt.tight_layout()

    if CPV is not None:
        fname = strjoin(AWARD_PROCEDURE, CPV, MODEL)
    else:
        fname = strjoin(AWARD_PROCEDURE, MODEL)
    fname = path.join(outpath, fname + "_scatter.png")
    plt.savefig(fname)
    plt.close(fig)


if __name__ == "__main__":

    # load global dataset
    df = prep.load_dataset(prep.INPUTDIR, prep.LOTTI_FNAME,
                           prep.VINCITORI_FNAME)
    df = prep.split_sum_totals(df)
    df = prep.feature_extraction(df)
    # df = prep.remove_obvious_outliers(df)
    df = prep.mark_outliers(df)
    df = df.rename(columns={"importo": "amount"})

    # load model results
    if CPV is not None:
        inpath = path.join(OUTDIR, AWARD_PROCEDURE, CPV, MODEL)
        fname = strjoin(AWARD_PROCEDURE, CPV, MODEL, SPLIT, "score.csv")
    else:
        inpath = path.join(OUTDIR, AWARD_PROCEDURE, MODEL)
        fname = strjoin(AWARD_PROCEDURE, MODEL, SPLIT, "score.csv")
    results = pd.read_csv(path.join(inpath, fname), index_col="idx")

    # extend results with global dataset features
    resultsExt = results.join(df, how="inner")

    # scatter results
    features = ["pa_med_ann_expenditure", "be_med_ann_revenue", "duration"]
    scatterscore(resultsExt, inpath, features)

    # plot outliers
    # plot_outlier_distr(N_OUTLIERS, resultsExt, df, inpath)
