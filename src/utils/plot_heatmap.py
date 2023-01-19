import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

DATADIR = "../../data10"
OUTDIR = "../../images/corr_heatmap/"


def heatmap(features, correlation_values):
    fig, ax = plt.subplots()

    mx = np.ma.masked_array(
        correlation_values, mask=np.triu(correlation_values, k=1))
    _ = ax.imshow(mx)
    # _ = ax.imshow(corrlation_values)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(features)),
                  labels=features, fontsize="xx-small")
    ax.set_yticks(np.arange(len(features)),
                  labels=features, fontsize="xx-small")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(features)):
        for j in range(len(features)):
            _ = ax.text(j, i, np.round(mx[i, j], 2),
                        ha="center", va="center", color="w", fontsize="xx-small")

    ax.set_title("Correlation")
    fig.tight_layout()
    return fig


def main():
    # load data
    df = pd.read_csv(DATADIR + "/contracts.csv", index_col="index")
    # select features to correlate
    features_to_one_hot = [
        'id_award_procedure',
        'start_date', 'id_lsf', 'id_forma_giuridica',
        'uber_forma_giuridica', 'cpv'
    ]
    features = [
        'amount', 'n_winners',
        'pa_med_ann_expenditure', 'be_med_ann_revenue', 'duration', 'be_std',
        'be_skewness', 'be_kurtosis', 'pa_std', 'pa_skewness', 'pa_kurtosis'
    ]
    # compute correlation
    X = df[features].corr().values
    # plot heatmap
    fig = heatmap(features, X)
    outpath = os.path.join(OUTDIR, "feature_correlation.png")
    fig.savefig(outpath, dpi=200)


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     # load dataset assuming there will be more than one tomorrow
#     for fname in os.listdir(DATADIR):

#         dataset = pd.read_csv(os.path.join(DATADIR, fname), index_col="idx")

#         # select the features you want to compute the correlation of
#         features = [
#             "pa_med_ann_expenditure", "be_med_ann_revenue", "duration",
#             "pa_med_ann_contr", "be_med_ann_contr",
#             "pa_med_ann_n_contr", "be_med_ann_n_contr"
#         ]

#         features = [
#             "pa_med_ann_expenditure", "be_med_ann_revenue", "duration",
#             'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7',
#             'month_8', 'month_9', 'month_10', 'month_11', 'month_12'
#         ]

#         features = [
#             "pa_med_ann_expenditure", "be_med_ann_revenue", "duration",
#             "sinMonth", "cosMonth"
#         ]

#         # compute corelation
#         X = dataset[features].corr().values

#         # plot
#         fig = heatmap(features, X)
#         # save plotted figure
#         award_procedure = fname[:-4]
#         outpath = os.path.join(OUTDIR, award_procedure)
#         plt.savefig(os.path.join(OUTDIR, "correlation.png"))
