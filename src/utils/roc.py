import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# def last_created_folder(path):
#     folders = glob.glob(path, recursive=True)
#     return max(folders, key=os.path.getctime)


fname = "output/020123-172509_aperta_oc-svm/aperta_oc-svm_test.csv"
ds = pd.read_csv(fname, index_col="index")

fpr, tpr, thr = roc_curve(ds["outlier"], ds["score"])

auc = roc_auc_score(ds["outlier"], ds["score"])

plt.plot(fpr, tpr)
plt.xlabel("False Alarm Rate")  # "False Positive Rate"
plt.ylabel("Hit Rate")  # True Positive Rate


# outpath = "images/roc/gmm/" + "test.png"
# plt.savefig(outpath)
