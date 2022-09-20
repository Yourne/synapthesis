from os import path
import os
import pandas as pd
import prep
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # load dataset - model results
    MODEL = "kde"
    DATASET_NAME = "appMed_aperta"
    OUTPUT_DIRECTORY = "output"
    N_OUTLIERS = 5
    output_path = path.join(OUTPUT_DIRECTORY, DATASET_NAME, MODEL)
    model_dataset_fname = MODEL + "-" + DATASET_NAME
    results_path = path.join(OUTPUT_DIRECTORY, model_dataset_fname + ".csv")
    results = pd.read_csv(results_path, index_col="idx")
    # load main dataset
    df = prep.load_dataset(prep.import_directory, prep.lotti_fn,
                           prep.vincitori_fn)
    df = prep.split_sum_totals(df)
    df = prep.feature_extraction(df)
    df = prep.remove_obvious_outliers(df)
    df = df.rename(columns={"importo": "amount"})
    # join dataframes
    dataset = results.join(df, how="inner")
    dataset = dataset.sort_values(by="scores")
    # for each outlier of the top 5 outliers:
    for i in range(N_OUTLIERS):
        # they are output sorted by score
        pa = int(dataset.iloc[i].id_pa)
        be = int(dataset.iloc[i].id_be)
        score = np.round(dataset.iloc[i].scores, 4)
        lot_id = str(int(dataset.iloc[i].id_lotto))
        outpath = path.join(output_path, lot_id)
        lot_log_amount = dataset.iloc[i].amount
        try:
            os.makedirs(outpath)
        except FileExistsError:
            pass
        # plot be and pa local and global amount distributions
        for scope, ds in {"local": dataset, "global": df}.items():
            for agent, agent_id in {"pa": pa, "be": be}.items():
                log_amount = np.log10(ds[ds["id_" + agent] == agent_id].amount)
                _, ax = plt.subplots()
                ax.set_xscale("log")
                ax.hist(log_amount)
                ax.vlines(lot_log_amount, 0, 1,
                          transform=ax.get_xaxis_transform(), colors="r")
                ax.set_title(agent + ": " + str(agent_id) + " score: " +
                             str(score))
                ax.set_xlabel("lot value")
                ax.set_ylabel("lot count")
                figure_name = agent + "-" + scope + "-" + "amount_distribution"
                fname = path.join(outpath, figure_name + ".png")
                plt.savefig(fname)
