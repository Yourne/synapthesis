# -*- coding: utf-8 -*-
import pandas as pd
from scipy import stats
import numpy as np


def shapiro_wilk_test(s: pd.Series) -> float:
    try:
        _, pvalue = stats.shapiro(s)
        return pvalue
    except ValueError:
        return 0


def boxcox(s: pd.Series) -> pd.Series:
    try:
        transformed_series, _ = stats.boxcox(s)
        return transformed_series
    except ValueError:
        return s


input_file = "../data10/contracts.csv"
output_path = "../data10/"

df = pd.read_csv(input_file, index_col="index")

features = ["id_be", "id_pa", "amount", "id_award_procedure", "duration"]

df = df[features]

grouped = df.groupby("id_be").amount
# pvalues = grouped.agg(shapiro_wilk_test)

# print(sum(pvalues > 0.05), sum(pvalues > 0.05) / len(pvalues))
# be sum(pvalues < 0.05) : 10305  # I reject most of the H_0

df["boxcox_transformed_amount"] = grouped.transform(boxcox)

for contracting_party in ["id_be", "id_pa"]:

    grouped = df.groupby(contracting_party).boxcox_transformed_amount

    pvalues = grouped.agg(shapiro_wilk_test)
    normally_distributed = (pvalues > 0.05).rename(
        contracting_party[3:] + "_normally_distributed")

    print(f"CONTRACTING PARTY {contracting_party}")
    print(sum(pvalues > 0.05), sum(pvalues > 0.05) / len(pvalues))

    mean = grouped.mean().rename(contracting_party[3:] + "_mean")
    std = grouped.std().rename(contracting_party[3:] + "_std")

    distribution_parameters = pd.concat(
        [mean, std, normally_distributed], axis=1)

    df = df.join(distribution_parameters, on=contracting_party)

    # if normally distribtued
    table = df[df[contracting_party[3:]+"_normally_distributed"] == True]
    resTrue = np.abs(table.boxcox_transformed_amount -
                     table[contracting_party[3:]+"_mean"]) > 3 * table[contracting_party[3:] + "_std"]

    # else
    table = df[df[contracting_party[3:]+"_normally_distributed"] == False]
    resFalse = np.abs(table.boxcox_transformed_amount -
                      table[contracting_party[3:]+"_mean"]) > 10 * table[contracting_party[3:] + "_std"]

    extreme_flag = pd.concat([resTrue, resFalse]).sort_index()

    extreme_flag = extreme_flag.rename(
        contracting_party[3:] + "_amount_extreme")
    extreme_flag.to_csv(
        output_path + contracting_party[3:] + "_amount_extreme.csv", index_label="index")

# DURATION
df = df.replace({"duration": 0}, 1)
grouped = df.groupby("id_award_procedure").duration
df["boxcox_transformed_duration"] = grouped.transform(boxcox)

grouped = df.groupby("id_award_procedure").boxcox_transformed_duration
pvalues = grouped.agg(shapiro_wilk_test)
print(sum(pvalues > 0.05), sum(pvalues > 0.05) / len(pvalues))
# the only normally distributed are procedure 30, 31, 33, none of our interest

mean = grouped.mean().rename("duration_mean")
std = grouped.std().rename("duration_std")
duration_parameters = pd.concat([mean, std], axis=1)
df = df.join(duration_parameters, on="id_award_procedure")
extreme_duration_flag = np.abs(
    df.boxcox_transformed_duration - df.duration_mean) > 10 * df.duration_std
extreme_duration_flag = extreme_duration_flag.rename("duration_extreme")
extreme_duration_flag.to_csv(
    output_path + "duration_extreme.csv", index_label="index")
