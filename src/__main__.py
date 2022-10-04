import importlib
import os
import pandas as pd
import time
from utils.utils import strjoin
# constants
# might change them to command-line arguments
MODEL = "kde"
OUTDIR = "output"

config = {
    "kde": {
        "module": "models.kde",
        "class": "KernelDensityEstimator",
    },
    "oc-svm": {
        "module": "models.oc_svm",
        "class": "OneClassSVM"
    }
}

MODEL = "kde"
config = config[MODEL]
module = importlib.import_module(config["module"])
model = getattr(module, config["class"])()


def recordexper(dsname):
    ts = time.strftime("%d%m%y-%H%M%S")
    folder = strjoin(ts, dsname, MODEL)
    outpath = os.path.join(OUTDIR, folder)
    try:
        os.makedirs(outpath)
    except FileExistsError:
        pass
    fname = os.path.join(outpath, dsname + "_" + MODEL)
    with open(fname + ".json", "w") as fp:
        fp.write(model.config())

    features = ["id_lotto", "score", "amount", "duration", "data_inizio", 
                "id_scelta_contraente", "id_pa", "uber_forma_giuridica",
                "id_be", "oggetto"]
    dataset[features].sort_values().to_csv(fname+".csv")


# laod a dataset
fname = "aperta.csv"
dataset = pd.read_csv(os.path.join("data", fname), index_col="idx")
X = model.preprocess(dataset)
# model.opt_params(config["optimizer"]["args"]["params"], X)
model.opt_params(X)
dataset["score"] = model.test(X)

recordexper("aperta")
