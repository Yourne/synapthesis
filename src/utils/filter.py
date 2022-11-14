import pandas as pd
import os
from scipy import stats
import numpy as np
import csv

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


def stdconfint(df: pd.DataFrame, alpha=.05) -> pd.DataFrame:
    n = len(df)
    std = np.std(df["amount"], ddof=1)
    upper = np.sqrt((n - 1) / stats.chi2.ppf(alpha / 2, n - 1)) * std
    lower = np.sqrt((n - 1) / stats.chi2.ppf((1 - alpha) / 2, n - 1)) * std
    return round(upper - lower, 4)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(INPUTDIR, FNAME))
    df["start_date"] = pd.to_datetime(df["start_date"])

    # BE
    entity = "id_be"
    ent = entity.strip("id_")

    # filter out the entities causing the boxcox to fail
    df = remove_boxcox_exceptions(df, entity)

    # trasnform the amount feature by IQR scaling and boxcox
    df[ent+"_amount"] = df["amount"].transform(IQRscale)
    df[ent+"_amount"] = df.groupby(entity)[ent+"_amount"].transform(boxcox)

    # compute 3 * standard deviations
    threestd = df.groupby(entity)[ent+"_amount"].std().rename(ent + "_3std")
    threestd = 3 * threestd
    df = df.join(threestd, on=entity)

    # compute the std 95 symmetric confidence interval
    confint = df.groupby(entity).apply(stdconfint)
    df = df.join(confint.rename(ent+"_stdconfint"), on=entity)

    # compute the mean
    mean = df.groupby(entity)[ent+"_amount"].mean()
    df = df.join(mean.rename(ent+"_mean"), on=entity)

    # Chebishev inequlity
    mask = np.abs(df[ent+"_amount"] - df[ent + "_mean"]) > df[ent+"_3std"]
    df[ent + "_probOut"] = mask

    # PA
    entity = "id_pa"
    ent = entity.strip("id_")

    # filter out the entities causing the boxcox to fail
    df = remove_boxcox_exceptions(df, entity)

    # trasnform the amount feature by IQR scaling and boxcox
    df["amount"] = df["amount"].transform(IQRscale)
    df["amount"] = df.groupby(entity)["amount"].transform(boxcox)

    # compute 3 * standard deviations
    threestd = df.groupby(entity)["amount"].std().rename(ent + "_3std")
    threestd = 3 * threestd
    df = df.join(threestd, on=entity)

    # compute the std 95 symmetric confidence interval
    confint = df.groupby(entity).apply(stdconfint)
    df = df.join(confint.rename(ent+"_stdconfint"), on=entity)

    # compute the mean
    mean = df.groupby(entity)["amount"].mean()
    df = df.join(mean.rename(ent+"_mean"), on=entity)

    # Chebishev inequlity
    mask = np.abs(df["amount"] - df[ent + "_mean"]) > df[ent+"_3std"]
    df[ent + "_probOut"] = mask

    # DURATION
    # replace null duration with 1
    df = df.replace({"duration": 0}, 1)
    # IQR scaling
    df["duration"] = df["duration"].transform(IQRscale)
    # boxcox
    df["duration"] = boxcox(df["duration"])
    # compute mean and std
    threestd = 3 * df["duration"].std()
    mean = df["duration"].mean()
    # Chebishev inquality
    mask = np.abs(df["duration"] - mean) > threestd
    df["duration_probOut"] = mask

    # sample dataset
    subset = df.groupby(["id_award_procedure", df.start_date.dt.year],
                        group_keys=False).apply(lambda x: x.sample(50,
                                                random_state=42))

    # save as csv
    features_to_csv = [
        "id_lotto", "amount", "duration", "start_date", "id_award_procedure",
        "id_pa", "uber_forma_giuridica", "id_be", "object",
        "be_med_ann_revenue", "pa_med_ann_expenditure", "be_probOut",
        "pa_probOut", "duration_probOut"
        ]
    fname = os.path.join(INPUTDIR, "subset.csv")
    subset[features_to_csv].to_csv(fname, index_label=False, index=False,
                                   quoting=csv.QUOTE_NONNUMERIC)
