# -*- coding: utf-8 -*-
import pandas as pd
import os

FNAME = "contracts.csv"  # do not use other files
INPUTDIR = "../../data10"


def rule1(df: pd.DataFrame) -> pd.Series:
    """
    If a contract lot has a value that exceeds the business entity median
    annual specific revenue and the contracting authority median annual
    expenditure, then it is an outlier
    """
    mask = (df["amount"] > df["be_med_ann_revenue"]) & \
        (df["amount"] > df["pa_med_ann_expenditure"])
    return mask
    # df = df[mask].copy()
    # df.loc[:, "rule"] = 1
    # return df


def rule2(df: pd.DataFrame) -> pd.Series:
    """
    If a contract lot is either a direct assignment or assignment under
    framework agreement and its duration is longer than ten years, then it is
    an outlier.
    """
    n_years = 10
    mask1 = (df["id_award_procedure"] == 23) & (df["duration"] > n_years * 365)
    mask2 = (df["id_award_procedure"] == 26) & (df["duration"] > n_years * 365)
    mask = mask1 | mask2
    return mask
    # df = df[mask].copy()
    # df["rule"] = 2
    # return df


def rule3(df: pd.DataFrame) -> pd.Series:
    """
    If a contract lot is 25 times bigger than the median annual specific rev-
    enue of the business entity that won the lot, then the contract lot is an
    outlier.
    """
    k = 25
    mask = df["amount"] > k * df["be_med_ann_revenue"]
    return mask
    # df = df[mask].copy()
    # df["rule"] = 3
    # return df


# def main_subset():
#     df = pd.read_csv(os.path.join(INPUTDIR, "subset_aperta.csv"))
#     df["start_date"] = pd.to_datetime(df["start_date"])
#     # rule 1
#     rule1 = (df["amount"] > df["be_med_ann_revenue"]) & \
#         (df["amount"] > df["pa_med_ann_expenditure"])
#     # rule 3
#     k = 25
#     rule3 = df["amount"] > k * df["be_med_ann_revenue"]

#     df["pred"] = False
#     df[rule1 | rule3] = True
#     return df

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(INPUTDIR, FNAME), index_col="index")
    df["start_date"] = pd.to_datetime(df["start_date"])
    out1 = rule1(df).rename("rule1")
    out1.to_csv(os.path.join(INPUTDIR, "rule1.csv"), index_label="index")

    out2 = rule2(df).rename("rule2")
    out2.to_csv(os.path.join(INPUTDIR, "rule2.csv"), index_label="index")

    out3 = rule3(df).rename("rule3")
    out3.to_csv(os.path.join(INPUTDIR, "rule3.csv"), index_label="index")

    # pd.concat([out1, out2, out3], axis=0)
