from os import path
import os
import pandas as pd
import prep
import matplotlib.pyplot as plt
import numpy as np
from utils import strjoin

OUTDIR = "output"
N_OUTLIERS = 5


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
    _, award_procedure, model, cpv = outpath.split("/")
    fig.suptitle(f"{model} {award_procedure} {cpv}")
    for i, feature in enumerate(features):
        s = ax[i].scatter(x=X[feature], y=X.amount, c=X.score, alpha=1, s=.2)
        ax[i].legend(*s.legend_elements())
        ax[i].set_ylabel("lot amount")
        ax[i].set_xlabel(feature)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
    plt.tight_layout()

    # save figure
    try:
        os.makedirs(outpath)
    except FileExistsError:
        pass

    fname = os.path.join(outpath, strjoin("_", [model, award_procedure, cpv,
                                                "scatter.png"]))
    plt.savefig(fname)
    plt.close(fig)


if __name__ == "__main__":

    # load global dataset
    df = prep.load_dataset(prep.import_directory, prep.lotti_fn,
                           prep.vincitori_fn)
    df = prep.split_sum_totals(df)
    df = prep.feature_extraction(df)
    df = prep.remove_obvious_outliers(df)
    df = df.rename(columns={"importo": "amount"})

    # for each local dataset
    for award_procedure in os.listdir(OUTDIR):
        path_to_scores = os.path.join(OUTDIR, award_procedure, "score")
        for fname in os.listdir(path_to_scores):
            if fname[-4:] == ".csv":
                model, award_procedure, cpv, _ = fname[:-4].split("_")

                # load local dataset
                results = pd.read_csv(os.path.join(path_to_scores, fname),
                                      index_col="idx")
                # extend results with df features
                resultsExt = results.join(df, how="inner")

                outpath = os.path.join(OUTDIR, award_procedure, model, cpv)
                # scatter results
                features = [
                    "pa_med_ann_expenditure", "be_med_ann_revenue", "duration"
                ]
                scatterscore(resultsExt, outpath, features)
                # plot outliers
                plot_outlier_distr(N_OUTLIERS, resultsExt, df, outpath)
