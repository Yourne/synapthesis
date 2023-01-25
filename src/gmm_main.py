import importlib
import pandas as pd
import numpy as np
# from src.utils import utils
import json
import matplotlib.pyplot as plt
import time


from sklearn.metrics import roc_curve, roc_auc_score


# constants
# might change them to command-line arguments
MODEL = "gmm"
OUTDIR = "output"

config = json.load(open("models/models_config.json"))

config = config[MODEL]
module = importlib.import_module(config["module"])
model = getattr(module, config["class"])()


# laod training set and validation set
X_train = pd.read_csv("../data10/train_test_open/X_train.csv")
X_test = pd.read_csv("../data10/train_test_open/X_test.csv")
y_test = pd.read_csv("../data10/train_test_open/y_test.csv")
y_train = pd.read_csv("../data10/train_test_open/y_train.csv")

features = ['be_duration', 'pa_duration',
            "be_duration_mean", "pa_duration_mean",
            "be_duration_std", "pa_duration_std",
            "be_duration_skewness", "pa_duration_skewness",
            "be_duration_kurtosis", "pa_duration_kurtosis",
            'be_amount', 'pa_amount',
            'be_amount_mean', 'pa_amount_mean',
            'be_amount_std', 'pa_amount_std',
            'be_amount_skewness', 'pa_amount_skewness',
            'be_amount_kurtosis', 'pa_amount_kurtosis',
            "be_med_ann_revenue", "pa_med_ann_expenditure",
            'n_winners']
X_train = X_train[features]
X_test = X_test[features]

# preprocess the model with the whole dataset
# X = model.preprocess(ds)

# model.fit(X_train)

# ds["score"] = model.test(X)
# subset = subset.join(ds["score"])
# subset["score"] = model.test(X_test)
# utils.recordexper("aperta", OUTDIR, MODEL, ds, model, subset)


# only for gmm
# features = ["be_amount", "pa_amount", "be_duration", "pa_duration"]
# performa meglio con più features
print("starting optimization")
start = time.time()
model.opt_params(X_train.values)
print(f"optimization elapsed {time.time() - start}")
print("best model")
print(model.model.__dict__)

bic = np.array(model.bic)
bic = np.reshape(bic, (-1, 4), order="F")
lines = plt.plot(model.n_components_range, bic)
plt.legend(lines, ["spherical", "tied", "diag", "full"])
plt.ylabel("Bayesian information criterion")
plt.xlabel("number of components")
plt.show()

preds = model.test(X_test.values)
# preds = model.test(X_train.values)

for outlier_label in ["extreme_amount", "extreme_duration", "rule_amount"]:
    fpr, tpr, thr = roc_curve(y_test[outlier_label], preds)
    auc = roc_auc_score(y_test[outlier_label], preds)
    # fpr, tpr, thr = roc_curve(y_train[outlier_label], preds)
    plt.plot(fpr, tpr, label=outlier_label)
    print(features)
    print(f"{outlier_label} auc {auc}")
plt.plot([0, 1], [0, 1], color="grey", zorder=1)
plt.grid()
plt.legend()
plt.xlabel("False Alarm Rate")  # "False Positive Rate"
plt.ylabel("Hit Rate")  # True Positive Rate
