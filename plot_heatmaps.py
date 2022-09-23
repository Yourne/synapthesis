import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

DATADIR = "data"
OUTDIR = "dataout"


def heatmap(features, corrlation_values):
    fig, ax = plt.subplots(figsize=(6.4*2, 4.3*2))
    _ = ax.imshow(corrlation_values)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(features)), labels=features)
    ax.set_yticks(np.arange(len(features)), labels=features)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(features)):
        for j in range(len(features)):
            _ = ax.text(j, i, np.round(corrlation_values[i, j], 2),
                        ha="center", va="center", color="w")

    ax.set_title("features correlation")
    fig.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "correlation.png"))


if __name__ == "__main__":
    # load dataset assuming there will be more than one tomorrow
    for fname in os.listdir(DATADIR):
        dataset = pd.read_csv(os.path.join(DATADIR, fname), index_col="idx")

        # select the features you want to compute the correlation of
        features = [
            "pa_med_ann_expenditure", "be_med_ann_revenue", "duration",
            "pa_med_ann_contr", "be_med_ann_contr",
            "pa_med_ann_n_contr", "be_med_ann_n_contr"
        ]
        
        # compute correlation
        X = dataset[features].corr().values
        
        # plot correlation
        # heatmap(features, X)
        # # print(dataset.columns)
        features = [
            "pa_med_ann_expenditure", "be_med_ann_revenue", "duration",
            'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7',
            'month_8', 'month_9', 'month_10', 'month_11', 'month_12'
        ]
        X = dataset[features].corr().values
        heatmap(features, X)
