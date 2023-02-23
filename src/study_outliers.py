#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:13:00 2023

@author: nepal
"""

import pandas as pd
# from scipy.stats import boxcox
import matplotlib.pyplot as plt
import seaborn as sns

input_path = "../data10/"
df = pd.read_csv("../data10/contracts.csv", index_col="index")

# load outliers baseline model
rule1 = pd.read_csv(input_path + "rule1.csv",
                    index_col="index")  # amount feature
rule2 = pd.read_csv(input_path + "rule2.csv",
                    index_col="index")  # duration feature
rule3 = pd.read_csv(input_path + "rule3.csv",
                    index_col="index")  # amount feature
# load outliers extreme value model
be_amount_extreme = pd.read_csv(
    input_path + "be_amount_extreme.csv", index_col="index")
pa_amount_extreme = pd.read_csv(
    input_path + "pa_amount_extreme.csv", index_col="index")
be_duration_extreme = pd.read_csv(
    input_path + "be_duration_extreme.csv", index_col="index")
pa_duration_extreme = pd.read_csv(
    input_path + "pa_duration_extreme.csv", index_col="index")

df = pd.concat([df, rule1, rule2, rule3, be_amount_extreme,
               pa_amount_extreme, be_duration_extreme, pa_duration_extreme], axis=1)

aperta = df[df["id_award_procedure"] == 1]

# plt.hist(aperta[aperta.rule1 == True].amount, bins=100)
be_outlier = list(set(aperta[aperta.rule1 == True].id_be))

subset = df[df.id_be.isin(be_outlier)]


amount_outliers = subset[subset.rule1 == True]
# fig, ax = plt.subplots()
# ax.hist(subset.amount, bins=10)
# _, ymax = ax.get_ylim()
# # ax.vlines(amount_outliers, 0, ymax, color="red", linestyles="dashed")
# ax.stem(amount_outliers, ymax)
# ax.scatter(amount_outliers, 0, color="red")
sns.boxplot(data=subset, x="id_be", y="amount")
