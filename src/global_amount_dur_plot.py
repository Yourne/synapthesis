#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:37:22 2023

@author: nepal
"""

import pandas as pd
from scipy.stats import boxcox
import matplotlib.pyplot as plt


def normalize(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min())


def my_boxcox(s: pd.Series) -> pd.Series:
    try:
        transformed_series, _ = boxcox(s)
        return transformed_series
    except ValueError:
        return s


df = pd.read_csv("../data10/contracts.csv", index_col="index")

party = "id_be"
feature = "amount"
grouped = df.groupby(party)
# normalized_feature = grouped[feature].transform(normalize)
# boxcoxed_feature, _ = boxcox(normalized_feature)
# plt.hist(boxcoxed_feature, bins=100)
t = grouped[feature].transform(normalize).transform(
    my_boxcox).transform(normalize)

plt.hist(t, bins=100)

party = "id_be"
feature = "duration"
grouped = df.groupby(party)
# normalized_feature = grouped[feature].transform(normalize)
# boxcoxed_feature, _ = boxcox(normalized_feature)
# plt.hist(boxcoxed_feature, bins=100)
t = grouped[feature].transform(normalize).transform(
    my_boxcox).transform(normalize)

plt.hist(t, bins=100)


party = "id_pa"
feature = "amount"
grouped = df.groupby(party)
# normalized_feature = grouped[feature].transform(normalize)
# boxcoxed_feature, _ = boxcox(normalized_feature)
# plt.hist(boxcoxed_feature, bins=100)
t = grouped[feature].transform(normalize).transform(
    my_boxcox).transform(normalize)

plt.hist(t, bins=100)


party = "id_pa"
feature = "duration"
grouped = df.groupby(party)
# normalized_feature = grouped[feature].transform(normalize)
# boxcoxed_feature, _ = boxcox(normalized_feature)
# plt.hist(boxcoxed_feature, bins=100)
t = grouped[feature].transform(normalize).transform(
    my_boxcox).transform(normalize)

plt.hist(t, bins=100)
