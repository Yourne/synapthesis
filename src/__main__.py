import importlib
import os
import pandas as pd
# import numpy as np
import utils.utils as utils
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


# laod a dataset
fname = "aperta"
dataset = pd.read_csv(os.path.join("data", fname+".csv"))
X = model.preprocess(dataset)
model.opt_params(X)
dataset["score"] = model.test(X)

utils.recordexper(fname, OUTDIR, MODEL, dataset, model)

# only for gmm
# bic = np.array(model.bic)
# bic = np.reshape(bic, (-1, 4), order="F")
# lines = plt.plot(model.n_components_range, bic)
# plt.legend(lines, ["spherical", "tied", "diag", "full"])
# plt.ylabel("Bayesian information criterion")
# plt.xlabel("number of components")
# plt.show()
