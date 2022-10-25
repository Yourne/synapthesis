import importlib
import os
import pandas as pd
import utils.utils as utils
import json

# constants
# might change them to command-line arguments
MODEL = "kde"
OUTDIR = "output"


fp = os.path.join("src", "models", "models_config.json")
config = json.load(open(fp))

config = config[MODEL]
module = importlib.import_module(config["module"])
model = getattr(module, config["class"])()


# laod a dataset
fname = "aperta"
dataset = pd.read_csv(os.path.join("data", fname+".csv"), index_col="idx")
X = model.preprocess(dataset)
# model.opt_params(config["optimizer"]["args"]["params"], X)
model.opt_params(X)
dataset["score"] = model.test(X)

utils.recordexper(fname, OUTDIR, MODEL, dataset, model)

