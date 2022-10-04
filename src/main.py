import os
import pandas as pd
from models import kde_main
from utils import utils

DATADIR = "data"
OUTDIR = "dataout"


if __name__ == "__main__":
    # TRAIN MODEL
    split = "train"
    award_procedure = "aperta"
    fname = award_procedure + "_" + split + ".csv"

    # load dataset
    dataset = pd.read_csv(os.path.join(DATADIR, fname), index_col="idx")
    X = kde_main.preprocess(dataset)
    model = kde_main.train(X)

    # TEST MODEL
    split = "test"
    fname = award_procedure + "_" + split + ".csv"
    fname = os.path.join(DATADIR, fname)
    # load test set
    testset = pd.read_csv(fname, index_col="idx")
    # preprocess
    X_te = kde_main.preprocess(testset)
    loglike = kde_main.test(model, X_te)

    # SAVE RESULTS
    # append scores to unprocessed dataset
    testset["score"] = loglike
    testset = testset.sort_values(by="score")

    # select saving position
    outpath = os.path.join(OUTDIR, award_procedure, kde_main.MODEL)

    # create folders if needed
    try:
        os.makedirs(outpath)
    except FileExistsError:
        pass

    # create filename
    fname = utils.strjoin(award_procedure, kde_main.MODEL, split, "score.csv")
    fname = os.path.join(outpath, fname)

    features = [
        "idx", "id_lotto", "importo", "duration", "score",
        "id_scelta_contraente", "data_inizio", "cpv"
    ]
    # dataset[features].to_csv(fname)

    # save the files for Davide
    # dataset.iloc[:100].to_csv(fname)
