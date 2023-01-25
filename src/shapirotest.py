# -*- coding: utf-8 -*-
import pandas as pd
from scipy import stats
import numpy as np
import warnings


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


def normalize(x: pd.Series, offset=1) -> pd.Series:
    if np.allclose([x.iloc[0]] * len(x), x):
        warnings.warn("All the values in the series are equal")
        return [1.5] * len(x)
    return x / (max(x) - min(x)) + 1


input_file = "../data10/contracts.csv"
output_path = "../data10/"

df = pd.read_csv(input_file, index_col="index")
df = df.replace({"duration": 0}, 1)

features = ["id_be", "id_pa", "amount", "id_award_procedure", "duration"]

df = df[features]

# ent = "pa"
# grouped = df.groupby("id_" + ent).duration
# df[ent + "_boxcoxed_duration"] = grouped.transform(boxcox)
# grouped = df.groupby("id_"+ent)[ent + "_boxcoxed_duration"]
# pvalues = grouped.agg(shapiro_wilk_test)


for party in ["be", "pa"]:
    for feature in ["amount", "duration"]:
        grouped = df.groupby("id_"+party)[feature]
        df[party+"_boxcoxed_" +
            feature] = grouped.transform(normalize).transform(boxcox)

        grouped = df.groupby(
            "id_"+party)[party+"_boxcoxed_"+feature]
        pvalues = grouped.agg(shapiro_wilk_test)
        normally_distributed = (pvalues > 0.05).rename(
            party + "_" + feature + "_normally_distributed")

        print(f"{party}, {feature}")
        print(sum(pvalues > 0.05), sum(
            pvalues > 0.05) / len(pvalues) * 100, len(pvalues))

        mean = grouped.mean().rename(party + "_" + feature + "_mean")
        std = grouped.std().rename(party + "_" + feature + "_std")

        distribution_parameters = pd.concat(
            [mean, std, normally_distributed], axis=1)

        df = df.join(distribution_parameters, on="id_" + party)

        # if normally distribtued
        table = df[df[party+"_" + feature + "_normally_distributed"] == True]
        resTrue = np.abs(table[party+"_boxcoxed_"+feature] -
                         table[party+"_" + feature + "_mean"]) > 3 * table[party + "_" + feature + "_std"]

        # else
        table = df[df[party+"_" + feature + "_normally_distributed"] == False]
        if feature == "duration":
            k = 10
        else:
            k = 20
        resFalse = np.abs(table[party+"_boxcoxed_"+feature] -
                          table[party+"_" + feature + "_mean"]) > k * table[party + "_" + feature + "_std"]

        extreme_flag = pd.concat([resTrue, resFalse]).sort_index()

        extreme_flag = extreme_flag.rename(
            party + "_" + feature + "_extreme")
        extreme_flag.to_csv(
            output_path + party + "_" + feature + "_extreme.csv", index_label="index")

# DURATION
# df = df.replace({"duration": 0}, 1)
# grouped = df.groupby("id_award_procedure").duration
# df["boxcox_transformed_duration"] = grouped.transform(boxcox)

# grouped = df.groupby("id_award_procedure").boxcox_transformed_duration
# pvalues = grouped.agg(shapiro_wilk_test)
# print(sum(pvalues > 0.05), sum(pvalues > 0.05) / len(pvalues))
# # the only normally distributed are procedure 30, 31, 33, none of our interest

# mean = grouped.mean().rename("duration_mean")
# std = grouped.std().rename("duration_std")
# duration_parameters = pd.concat([mean, std], axis=1)
# df = df.join(duration_parameters, on="id_award_procedure")
# extreme_duration_flag = np.abs(
#     df.boxcox_transformed_duration - df.duration_mean) > 10 * df.duration_std
# extreme_duration_flag = extreme_duration_flag.rename("duration_extreme")
# extreme_duration_flag.to_csv(
#     output_path + "duration_extreme.csv", index_label="index")
