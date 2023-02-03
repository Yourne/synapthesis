import importlib
import pandas as pd
import numpy as np
# from src.utils import utils
import json
import matplotlib.pyplot as plt
import time
from datetime import datetime


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
# old: train test open : norm boxcox
# new: train_test_open_full: norm boxbox norm
X_train = pd.read_csv("../data10/train_test_open_full/X_train.csv")
X_test = pd.read_csv("../data10/train_test_open_full/X_test.csv")
y_test = pd.read_csv("../data10/train_test_open_full/y_test.csv")
y_train = pd.read_csv("../data10/train_test_open_full/y_train.csv")

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
model.opt_params(X_train.values)  # 107 s
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
plt.savefig("..images/gmm_bic60")

preds = model.test(X_test.values)


def record_experiment(preds: np.array, X_test: pd.DataFrame, model: str):
    date = datetime.now().isoformat()
    file_path = "../output/" + date + "_" + model + ".csv"
    preds_series = pd.Series(preds, index=X_test.index, name="prediction")
    preds_series.to_csv(file_path)


record_experiment(preds, X_test, "gmm")


# cannot visualize as it has 23 dims
# def draw_contuours(X: pd.DataFrame, y: pd.DataFrame, model):

#     xx = np.linspace(0, 1)
#     yy = np.linspace(0, 1)

#     YY, XX = np.meshgrid(xx, yy)
#     xy = np.vstack([XX.ravel(), YY.ravel()]).T

#     Z = model.test(xy).reshape(XX.shape)

#     CS = plt.contour(XX, YY, Z)
#     plt.clabel(CS)

#     for i in range(4):
#         X_outliers = X[y.iloc[:, i] == -1]
#         plt.scatter(X_outliers.iloc[:, 0],
#                     X_outliers.iloc[:, 1], label=y.columns[i])
#     plt.legend(title="Type of outliers")

# draw_contuours(X_test, y_test, model)


for outlier_label in ["extreme_amount", "extreme_duration", "rule_amount"]:
    fpr, tpr, thr = roc_curve(y_test[outlier_label], preds)
    auc = roc_auc_score(y_test[outlier_label], preds)
    # fpr, tpr, thr = roc_curve(y_train[outlier_label], preds)
    # plt.plot(fpr, tpr, label=outlier_label)
    print(f"{outlier_label} auc {auc}")
plt.plot([0, 1], [0, 1], color="grey", zorder=1)
plt.grid()
plt.legend()
plt.xlabel("False Alarm Rate")  # "False Positive Rate"
plt.ylabel("Hit Rate")  # True Positive Rate
plt.savefig("../images/roc/gmm/normed-gmm")
