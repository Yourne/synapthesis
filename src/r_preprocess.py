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


def normalize(x: pd.Series, offset=1) -> pd.Series:
    if np.allclose([x.iloc[0]] * len(x), x):
        warnings.warn("All the values in the series are equal")
        return [1.5] * len(x)
    return x / (max(x) - min(x)) + 1


def boxcox(s: pd.Series) -> pd.Series:
    try:
        transformed_series, _ = stats.boxcox(s)
        return transformed_series
    except ValueError:
        return s


def sample_skewness(s: pd.Series) -> float:
    n = len(s)
    # second sample moment
    m2 = sum((s - s.mean()) ** 2) / n
    if m2 == 0:
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
    if m2 == 0:
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


input_file = "../data10/contracts.csv"
df = pd.read_csv(input_file, index_col="index")
df["start_date"] = pd.to_datetime(df["start_date"])
features = ["id_be", "id_pa", "amount",
            "id_award_procedure", "duration", "start_date", "n_winners"]
df = df[features]


for party in ["be", "pa"]:
    for feature in ["amount", "duration"]:
        grouped = df.groupby("id_"+party)[feature]
        df[party+"_"+feature] = grouped.transform(normalize).transform(boxcox)
        df = compute_moments(df, party+"_"+feature, party)
        # std = df.groupby("id_"+party)[party + "_" +
        #                                           feature].std()
        # std = std.rename(party + "_" + feature + "_std")
        # df = df.join(std, on="id_"+party)
        if feature == "amount":
            df = extract_median_yearly_revenue(df, party)

# df = df.replace({"duration": 0}, 1)
# grouped = df.groupby("id_award_procedure").duration
# df["duration"] = grouped.transform(boxcox)

# load outlier
outlier_features = ["rule_amount", "rule_duration",
                    "extreme_amount", "extreme_duration"]
rule_amount = pd.read_csv("../data10/rule_amount.csv").squeeze()
rule_duration = pd.read_csv("../data10/rule_duration.csv").squeeze()
extreme_amount = pd.read_csv("../data10/extreme_amount.csv").squeeze()
extreme_duration = pd.read_csv("../data10/extreme_duration.csv").squeeze()
df["rule_amount"] = rule_amount
df["rule_duration"] = rule_duration
df["extreme_amount"] = extreme_amount
df["extreme_duration"] = extreme_duration

# output features
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

df[features].to_csv("../data10/processed_data.csv", index_label=False)

aperta = df[df.id_award_procedure == 1]
rng = np.random.default_rng(seed=1)
test_idx = rng.choice(aperta.index.values, size=round(len(aperta)*.3))
(aperta.loc[test_idx, outlier_features] == -1).sum()
train_idx = aperta.index.difference(test_idx)

outpath = "../data10/train_test_open/"
aperta.loc[train_idx, features].to_csv(
    outpath+"X_train.csv", index_label=False)
aperta.loc[test_idx, features].to_csv(outpath+"X_test.csv", index_label=False)
aperta.loc[train_idx, outlier_features].to_csv(
    outpath+"y_train.csv", index_label=False)
aperta.loc[test_idx, outlier_features].to_csv(
    outpath+"y_test.csv", index_label=False)
