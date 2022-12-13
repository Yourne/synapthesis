import pandas as pd
import os
from scipy import stats
import numpy as np
import csv

FNAME = "contracts.csv"
INPUTDIR = "data"
K = 2


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
        return df
    else:
        mask = df[entity] == exceptions[0]
        for e in exceptions[1:]:
            mask += df[entity] == e
        return df[~mask]


def IQRscale(s: pd.Series) -> pd.Series:
    # scale by interquartile range
    q25, q75 = s.quantile([.25, .75])
    return s / (q75 - q25)


def boxcox(s: pd.Series) -> np.array:
    return stats.boxcox(s)[0]


def stdconfint(df: pd.DataFrame, entity: str, alpha=.05) -> pd.DataFrame:
    n = len(df)
    std = np.std(df[ent+"_amount"], ddof=1)
    upper = np.sqrt((n - 1) / stats.chi2.ppf(alpha / 2, n - 1)) * std
    lower = np.sqrt((n - 1) / stats.chi2.ppf((1 - alpha) / 2, n - 1)) * std
    return round(upper - lower, 4)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(INPUTDIR, FNAME))
    df["start_date"] = pd.to_datetime(df["start_date"])

    for entity in ["id_be", "id_pa"]:
        ent = entity.strip("id_")

        # filter out the entities causing the boxcox to fail
        df = remove_boxcox_exceptions(df, entity)

        # trasnform the amount feature by IQR scaling and boxcox
        df[ent+"_amount"] = df["amount"].transform(IQRscale)
        df[ent+"_amount"] = df.groupby(entity)[ent+"_amount"].transform(boxcox)

        # compute 3 * standard deviations
        kstd = df.groupby(entity)[ent+"_amount"].std().rename(ent+"_kstd")
        kstd = K * kstd
        df = df.join(kstd, on=entity)

        # compute the number of won lots
        nlots = df.groupby(entity).size().rename(ent+"_nlots")
        df = df.join(nlots, on=entity)

        # compute the std
        std = df.groupby(entity)[ent+"_amount"].std().rename(ent+"_std")
        df = df.join(std, on=entity)

        # compute the mean
        mean = df.groupby(entity)[ent+"_amount"].mean()
        df = df.join(mean.rename(ent+"_mean"), on=entity)

        # Chebishev inequlity
        mask = np.abs(df[ent+"_amount"] - df[ent + "_mean"]) > df[ent+"_kstd"]
        df[ent + "_probOut"] = mask

    # DURATION
    # replace null duration with 1
    df = df.replace({"duration": 0}, 1)
    # IQR scaling
    df["duration_scaled"] = df["duration"].transform(IQRscale)
    # boxcox
    df["duration_scaled"] = boxcox(df["duration_scaled"])
    # compute mean and std
    kstd = K * df["duration_scaled"].std()
    mean = df["duration_scaled"].mean()
    # Chebishev inquality
    mask = np.abs(df["duration_scaled"] - mean) > kstd
    df["duration_probOut"] = mask

    aperta = df[df["id_award_procedure"] == 1]
    subset = aperta.groupby(
        df.start_date.dt.year,
        group_keys=False).apply(lambda x: x.sample(67, random_state=42))

    # save as csv
    features = [
        "amount",
        "be_med_ann_revenue", "be_probOut", "be_std", "be_nlots",
        "pa_med_ann_expenditure", "pa_probOut", "pa_std", "pa_nlots",
        "duration",
        "object", "start_date", "id_award_procedure", "id_pa", "id_be",
        "id_lotto"
    ]

    # true outliers lot ids checked on the platform
    outliers = [7531663]
    subset["outlier"] = False
    subset.loc[subset["id_lotto"] == outliers[0], "outlier"] = True
    features.append("outlier")

    fname = os.path.join(INPUTDIR, "subset_aperta.csv")
    subset[features].to_csv(fname, index_label=False, index=False,
                            quoting=csv.QUOTE_NONNUMERIC)
