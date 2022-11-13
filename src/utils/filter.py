import pandas as pd
import os
from scipy import stats
import numpy as np

FNAME = "contracts.csv"
INPUTDIR = "data"


def remove_boxcox_exceptions(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    exceptions = list()
    for name, group in df.groupby(entity):
        try:
            stats.boxcox(group["amount"])[0]
        except ValueError:
            exceptions.append(name)
            continue

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


def stdconfint(df: pd.DataFrame, alpha=.05) -> pd.DataFrame:
    n = len(df)
    std = np.std(df["amount"], ddof=1)
    upper = np.sqrt((n - 1) / stats.chi2.ppf(alpha / 2, n - 1)) * std
    lower = np.sqrt((n - 1) / stats.chi2.ppf((1 - alpha) / 2, n - 1)) * std
    return round(upper - lower, 4)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(INPUTDIR, FNAME), index_col="idx")

    for entity in ["id_be", "id_pa"]:
        entity = "id_be"
        ent = entity.strip("id_")

        # filter out the entities causing the boxcox to fail
        df = remove_boxcox_exceptions(df, entity)

        # trasnform the amount feature by IQR scaling and boxcox
        df["amount"] = df["amount"].transform(IQRscale)
        df["amount"] = df.groupby(entity)["amount"].transform(boxcox)

        # compute 3 * standard deviations
        threestd = df.groupby(entity)["amount"].std()
        threestd = 3 * threestd.rename(ent + "_3std")
        df = df.join(threestd, on=entity)

        # compute the std 95 symmetric confidence interval
        confint = df.groupby(entity).apply(stdconfint)
        df = df.join(confint.rename(ent+"_stdconfint"), on=entity)

        # compute the mean
        mean = df.groupby(entity)["amount"].mean()
        df = df.join(mean.rename(ent+"_mean"), on=entity)

        # Chebishev inequlity
        mask = np.abs(df["amount"] - df[ent + "_mean"]) > df[ent+"_3std"]
        df[ent + "_probableOutlier"] = mask

