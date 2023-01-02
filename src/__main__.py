import importlib
import os
import pandas as pd
import numpy as np
from src.utils import utils
import json
# import matplotlib.pyplot as plt


# constants
# might change them to command-line arguments
MODEL = "oc-svm"
OUTDIR = "output"


fp = os.path.join("src", "models", "models_config.json")
config = json.load(open(fp))

config = config[MODEL]
module = importlib.import_module(config["module"])
model = getattr(module, config["class"])()


# laod training set and validation set
ds = pd.read_csv("data/contracts.csv", index_col="index")
ds = ds[ds["id_award_procedure"] == 1]
subset = pd.read_csv("data/subset_aperta.csv", index_col="index")

# preprocess the model with the whole dataset
X = model.preprocess(ds)

# divide training and test set
# get indices of the subset in the dataset
idx = ds.index.get_indexer(subset.index)
X_train = np.delete(X, idx, axis=0)
X_test = X[idx, :]

model.fit(X_train)

ds["score"] = model.test(X)
# subset = subset.join(ds["score"])
subset["score"] = model.test(X_test)
utils.recordexper("aperta", OUTDIR, MODEL, ds, model, subset)


# only for gmm
# bic = np.array(model.bic)
# bic = np.reshape(bic, (-1, 4), order="F")
# lines = plt.plot(model.n_components_range, bic)
# plt.legend(lines, ["spherical", "tied", "diag", "full"])
# plt.ylabel("Bayesian information criterion")
# plt.xlabel("number of components")
# plt.show()
