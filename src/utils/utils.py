import time
import os
import csv


def strjoin(*words, separator="_"):
    s = words[0]
    for w in words[1:]:
        s += separator + w
    return s


def recordexper(DSNAME, OUTDIR, MODEL, dataset, model):
    ts = time.strftime("%d%m%y-%H%M%S")
    folder = strjoin(ts, DSNAME, MODEL)
    outpath = os.path.join(OUTDIR, folder)
    try:
        os.makedirs(outpath)
    except FileExistsError:
        pass
    fname = os.path.join(outpath, DSNAME + "_" + MODEL)
    with open(fname + ".json", "w") as fp:
        fp.write(model.config())

    features = ["id_lotto", "score", "amount", "duration", "start_date",
                "id_award_procedure", "id_pa", "uber_forma_giuridica",
                "id_be", "object", "be_med_ann_revenue",
                "pa_med_ann_expenditure"]
    dataset[features].sort_values("score").to_csv(
        fname+".csv", index_label=False, index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
