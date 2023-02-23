#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 12:24:42 2023

@author: nepal
"""

import pandas as pd
from scipy import stats
import numpy as np
import warnings
from sklearn.decomposition import PCA


def normalize(x: pd.Series, offset: float = 1) -> pd.Series:
    if np.allclose([x.iloc[0]] * len(x), x):
        warnings.warn("All the values in the series are equal")
        return [offset + .5] * len(x)
    return (x - min(x)) / (max(x) - min(x)) + offset


def boxcox(s: pd.Series) -> pd.Series:
    # se faccio sta cosa pd.transform dà "Series not transformed"
    # if np.allclose([s.iloc[0]] * len(s), s):
    #     warnings.warn("All the values in the series are equal")
    #     return s
    # else:
    #     transformed_series, _ = stats.boxcox(s)
    #     return transformed_series
    try:
        transformed_series, _ = stats.boxcox(s)
        return transformed_series
    except ValueError:
        return s


def sample_skewness(s: pd.Series) -> float:
    n = len(s)
    # second sample moment
    m2 = sum((s - s.mean()) ** 2) / n
    if np.isclose(m2, 0):
        return 0  # each contract has the same value
    # third sample moment
    m3 = sum((s - s.mean()) ** 3) / n
    # biased sample skewness
    g1 = m3 / (m2 ** (3/2))
    # unbiased sample skewness
    G1 = ((n * (n-1)) ** (1/2)) / (n - 2) * g1
    return G1


def sample_kurtosis(s: pd.Series) -> float:
    n = len(s)
    # second sample moment
    m2 = sum((s - s.mean()) ** 2) / n
    if np.isclose(m2, 0):
        return 0  # each contract has the same value
    # forth sample moment
    m4 = sum((s - s.mean()) ** 4) / n
    # biased sample kurtosis
    g2 = m4 / (m2 ** 2) - 3
    # unbiased sample kurtosis
    G2 = ((n-1) * ((n+1) * g2 + 6)) / ((n - 2) * (n - 3))
    return G2


def compute_moments(df: pd.DataFrame, feature: str, party: str) -> pd.DataFrame:
    grouped = df.groupby("id_"+party)[feature]
    mean = grouped.mean().rename(feature + "_mean")
    std = grouped.std().rename(feature + "_std")
    skewness = grouped.agg(sample_skewness)
    skewness = skewness.rename(feature + "_skewness")
    kurtosis = grouped.agg(sample_kurtosis)
    kurtosis = kurtosis.rename(feature + "_kurtosis")
    moments = pd.concat([mean, std, skewness, kurtosis], axis=1)
    df = df.join(moments, on="id_"+party)
    return df


def extract_median_yearly_revenue(df, agent):
    feat_name = agent + "_amount"
    rev_by_year = df.groupby(
        ["id_"+agent, df.start_date.dt.year])[feat_name].sum()
    rev_by_year = rev_by_year.unstack()
    med_yearly_rev = rev_by_year.median(axis=1)
    if agent == "pa":
        med_yearly_rev = med_yearly_rev.rename("pa_med_ann_expenditure")
    else:
        med_yearly_rev = med_yearly_rev.rename("be_med_ann_revenue")
    return df.join(med_yearly_rev, on="id_"+agent)


def encode_labels(df: pd.DataFrame, column: str) -> pd.DataFrame:
    labels_to_integer = dict()
    for i, label in enumerate(set(df[column])):
        labels_to_integer[label] = i
    return df.replace({column: labels_to_integer})


def shapiro_wilk_test(s: pd.Series) -> float:
    try:
        _, pvalue = stats.shapiro(s)
        return pvalue
    except ValueError:
        return 0


input_file = "../data10/contracts.csv"
df = pd.read_csv(input_file, index_col="index")
df["start_date"] = pd.to_datetime(df["start_date"])
# df = df.replace({"duration": 0}, 1)

categorical_features = ['id_lsf', 'id_forma_giuridica',
                        'uber_forma_giuridica', 'cpv', 'id_award_procedure']
df_categorical = df[categorical_features]
df_categorical = encode_labels(df_categorical, "uber_forma_giuridica")

features_to_process = ["id_be", "id_pa", "amount",
                       "id_award_procedure", "duration", "start_date", "n_winners"]
df = df[features_to_process]


for party in ["be", "pa"]:
    for feature in ["amount", "duration"]:
        print()
        print(party.upper(), feature.upper())
        grouped = df.groupby("id_"+party)[feature]
        df[party+"_"+feature] = grouped.transform(
            normalize).transform(boxcox).transform(normalize, offset=0)
        # pvalues = df.groupby(
        #     "id_"+party)[party+"_"+feature].agg(shapiro_wilk_test)
        # print(party, feature, sum(pvalues > 0.05) / len(pvalues))
        df = compute_moments(df, party+"_"+feature, party)
        if feature == "amount":
            df = extract_median_yearly_revenue(df, party)

# load outliers
outlier_features = ["rule_amount", "rule_duration",
                    "extreme_amount", "extreme_duration"]
rule_amount = pd.read_csv("../data10/rule_amount.csv").squeeze()
rule_duration = pd.read_csv("../data10/rule_duration.csv").squeeze()
extreme_amount = pd.read_csv("../data10/extreme_amount.csv").squeeze()
extreme_duration = pd.read_csv("../data10/extreme_duration.csv").squeeze()
y = pd.concat([rule_amount, rule_duration,
              extreme_amount, extreme_duration], axis=1)

processed_features = ['be_duration', 'pa_duration',
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

df = df[processed_features]
df = pd.concat([df, df_categorical], axis=1)

df_open = df[df.id_award_procedure == 1]
df_open = df_open.drop(columns=["id_award_procedure"])
rng = np.random.default_rng(seed=1)
test_idx = rng.choice(df_open.index.values, size=round(len(df_open)*.3),
                      replace=False)
(y.loc[test_idx, :] == -1).sum()
train_idx = df_open.index.difference(test_idx)

# if PCA
outpath = "../data10/train_test_open_PCA/"
pca = PCA()
X_transformed = pca.fit_transform(df)
X_transformed = pd.DataFrame(X_transformed, index=df.index)

X_transformed.loc[train_idx, :].to_csv(
    outpath+"X_train.csv", index_label=False)
X_transformed.loc[test_idx, :].to_csv(outpath+"X_test.csv", index_label=False)
y.loc[train_idx, :].to_csv(outpath+"y_train.csv", index_label=False)
y.loc[test_idx, :].to_csv(outpath+"y_test.csv", index_label=False)

# if not PCA
# outpath = "../data10/train_test_open_full/"
# df_open.loc[train_idx, :].to_csv(
#     outpath+"X_train.csv", index_label=False)
# df_open.loc[test_idx, :].to_csv(outpath+"X_test.csv", index_label=False)
# y.loc[train_idx, :].to_csv(
#     outpath+"y_train.csv", index_label=False)
# y.loc[test_idx, :].to_csv(
#     outpath+"y_test.csv", index_label=False)
