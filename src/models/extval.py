# -*- coding: utf-8 -*-

import pandas as pd
import os
from scipy import stats
import numpy as np

FNAME = "contracts.csv" # do not use other files
INPUTDIR = "data"
K = 3 # MODEL PARAMETER TO OPTIMIZE


def remove_boxcox_exceptions(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    exceptions = list()
    for name, group in df.groupby(entity):
        try:
            stats.boxcox(group["amount"])[0]
        except ValueError:
            exceptions.append(name)
            continue

    print(f"boxcox exceptions: {len(exceptions)}")

    if len(exceptions) == 0:
        return None, df
    else:
        mask = df[entity] == exceptions[0]
        for e in exceptions[1:]:
            mask += df[entity] == e
        return exceptions, df[~mask]


def IQRscale(s: pd.Series) -> pd.Series:
    # scale by interquartile range
    q25, q75 = s.quantile([.25, .75])
    return s / (q75 - q25)


def boxcox(s: pd.Series) -> np.array:
    return stats.boxcox(s)[0]


def stdconfint(df: pd.DataFrame, entity: str, alpha=.05) -> pd.DataFrame:
    n = len(df)
    std = np.std(df[entity+"_amount"], ddof=1)
    upper = np.sqrt((n - 1) / stats.chi2.ppf(alpha / 2, n - 1)) * std
    lower = np.sqrt((n - 1) / stats.chi2.ppf((1 - alpha) / 2, n - 1)) * std
    return round(upper - lower, 4)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(INPUTDIR, FNAME))
    df["start_date"] = pd.to_datetime(df["start_date"])

    for entity in ["id_be", "id_pa"]:
        ent = entity.strip("id_")

        # filter out the entities causing the boxcox to fail 
        exceptions, df = remove_boxcox_exceptions(df, entity)
        # exceptions are either entities with only one lot, either entities
        # with many lots with the same price.

        # trasnform the amount feature by IQR scaling and boxcox
        df[ent+"_amount"] = df["amount"].transform(IQRscale)
        df[ent+"_amount"] = df.groupby(entity)[ent+"_amount"].transform(boxcox)

        # compute the number of won lots
        nlots = df.groupby(entity).size().rename(ent+"_nlots")
        df = df.join(nlots, on=entity)

        # compute the K std
        std = df.groupby(entity)[ent+"_amount"].std().rename(ent+"_kstd")
        kstd = K * std
        df = df.join(kstd, on=entity)
        
        # compute the std
        std = df.groupby(entity)[ent+"_amount"].std().rename(ent+"_std")
        df = df.join(std, on=entity)

        # compute the mean
        mean = df.groupby(entity)[ent+"_amount"].mean()
        df = df.join(mean.rename(ent+"_mean"), on=entity)

        # Chebishev inequlity
        mask = np.abs(df[ent+"_amount"] - df[ent + "_mean"]) > df[ent+"_kstd"]
        df[ent + "_probOut"] = mask
        
        # compute z-score
        df[ent+"_z"] = (df[ent+"_amount"] - df[ent+"_mean"]) / df[ent+"_std"]
        df[ent+"_prob"] = df[ent+"_z"] ** (-2)
        

    # DURATION
    # replace null duration with 1
    df = df.replace({"duration": 0}, 1)
    # IQR scaling
    df["duration_scaled"] = df["duration"].transform(IQRscale)
    # boxcox
    df["duration_scaled"] = boxcox(df["duration_scaled"])
    # compute mean and std
    mean = df["duration_scaled"].mean()
    std = df["duration_scaled"].std()
    # Chebishev inquality
    mask = np.abs(df["duration_scaled"] - mean) > K * std
    df["duration_probOut"] = mask
    # compute z-score
    df["dur_z"] = (df["duration_scaled"] - mean) / std
    df["dur_prob"] = df["dur_z"] **(-2)

    # select features
    features = [
        "amount",
        "be_med_ann_revenue","be_prob", "be_amount", "be_mean", "be_std",
        "pa_med_ann_expenditure", "pa_prob", "pa_amount", "pa_mean", "pa_std",
        "duration", "dur_prob", "duration_scaled",
        "object", 
        "start_date", "id_award_procedure", "id_pa", "id_be", "id_lotto"
    ]
    df = df[features]

    # select open procedure
    df = df[df["id_award_procedure"] == 1]
    
    # optimize K on the subset 
    # will be done manually on google sheet
    
    # load the subset
    subset = df.groupby(
        df.start_date.dt.year,
        group_keys=False).apply(lambda x: x.sample(67, random_state=42))
    outliers = [7531663]
    subset["outlier"] = False
    subset.loc[subset["id_lotto"] == outliers[0], "outlier"] = True

    
    # csv to be load on google sheet
    df.to_csv("output/EV_aperta.csv")
    subset.to_csv("output/EV_aperta_subset.csv")
    

    


