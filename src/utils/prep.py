
import csv
from os import path
import numpy as np
import pandas as pd

OUTDIR = "data10"  # "data10" # "data_old"
INPUTDIR = "synData6July"
LOTTI_FNAME = "export_lotti_veneto_2016_2018_giulio_v2.csv"
VINCITORI_FNAME = "export_vincitori_veneto_2016_2018_giulio_v2.csv"


def replace_missing_value(df, col, replacement_col):
    mask = df[replacement_col].notna() & df[col].isna()
    df.loc[df[mask].index, col] = df[replacement_col][mask]
    return df


def load_dataset():
    lotti = pd.read_csv(path.join(INPUTDIR, LOTTI_FNAME))
    vincitori = pd.read_csv(path.join(INPUTDIR, VINCITORI_FNAME))
    print("lotti {}".format(lotti.shape))
    print("vincitori {}".format(vincitori.shape))

    # convert datatypes
    lotti.data_inizio = pd.to_datetime(lotti.data_inizio, yearfirst=True)
    lotti.data_fine = pd.to_datetime(lotti.data_fine, yearfirst=True)
    lotti.data_inferita = pd.to_datetime(lotti.data_inferita, yearfirst=True)

    # replace missing values in col1 with values from col2
    # lotti = replace_missing_value(lotti, "importo", "importo_base_asta")
    lotti = replace_missing_value(lotti, "data_inizio", "data_inferita")

    # drop table attributes with mostly missing values
    missingValuesCols = [
        "importo_liquidato", "importo_base_asta", "data_inferita",
        "id_mod_realizz", "cpv_vero"]
    lotti = lotti.drop(columns=missingValuesCols)
    # print("columns dropped with mostly missing values:")
    # print(str(missingValuesCols))

    # clean the dfs from the remaining missing values
    lotti = lotti.dropna()
    vincitori = vincitori.dropna()
    print(f"lotti after dropna {lotti.shape}")
    print(f"vincitori after dropna {vincitori.shape}")

    # print("dropped all the rows with at least one missing value")
    # cast to int64 cols now w/out np.nan
    lotti.id_scelta_contraente = lotti.id_scelta_contraente.astype('int')
    lotti.id_lsf = lotti.id_lsf.astype('int')
    lotti.cpv = lotti.cpv.astype('int')
    # drop columns that lead to sparse matrices
    # following tests: try models that handle sparse datasets
    # lotti = lotti.drop(
    #     columns=["id_forma_giuridica", "uber_forma_giuridica"])
    vincitori = vincitori.drop(
        columns=["id_forma_giuridica", "uber_forma_giuridica"])

    # merge the datasets
    df = lotti.merge(vincitori, on="id_lotto", how="inner")
    print(f"merged shape {df.shape}")
    return df


def split_sum_totals(df):
    n_winners = df.groupby("id_lotto").size().rename("n_winners")
    df = df.merge(n_winners, how="left", on="id_lotto")
    df["importo"] = df["importo"] / df["n_winners"]
    return df


def extract_median_yearly_revenue(df, agent):
    rev_by_year = df.groupby([agent, df.data_inizio.dt.year]).importo.sum()
    rev_by_year = rev_by_year.unstack()
    med_yearly_rev = rev_by_year.median(axis=1)
    if agent == "id_pa":
        med_yearly_rev = med_yearly_rev.rename("pa_med_ann_expenditure")
    else:
        med_yearly_rev = med_yearly_rev.rename("be_med_ann_revenue")
    return df.join(med_yearly_rev, on=agent)


def extract_median_contract(df, agent):
    # the median of the median of each year. why? for consistency with the
    # other extracted features
    med_year_contr = df.groupby([
        agent, df.data_inizio.dt.year]).median().importo
    med_year_contr = med_year_contr.unstack()
    med_year_contr = med_year_contr.median(axis=1)
    med_year_contr = med_year_contr.rename(
        agent.strip("id_") + "_med_ann_contr")
    return df.join(med_year_contr, on=agent)


def extract_median_yearly_n_contracts(df, agent):
    contr_by_year = df.groupby(
        [agent, df.data_inizio.map(lambda x: x.year)]).size()
    contr_by_year = contr_by_year.unstack()
    med_yearly_n_contr = contr_by_year.median(axis=1)
    med_yearly_n_contr = med_yearly_n_contr.rename(
        agent.strip("id_") + "_med_ann_n_contr")
    return df.join(med_yearly_n_contr, on=agent)


def encode_sin_cos(df, period="DayOfYear"):
    if period == "Month":
        x = df.data_inizio.dt.month
        period_items = 12
    else:
        x = df.data_inizio.dt.day_of_year
        period_items = 365
    df["sin" + period] = np.sin(x / period_items * 2 * np.pi)
    df["cos" + period] = np.cos(x / period_items * 2 * np.pi)
    return df


def dummy_month(df):
    dummies = pd.get_dummies(df.data_inizio.dt.month, prefix="month",
                             drop_first=True)
    return pd.concat([df, dummies], axis=1)


def encode_month_rbf(df):
    pass


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


def compute_moments(df: pd.DataFrame) -> pd.DataFrame:
    for party in ["id_be", "id_pa"]:
        grouped = df.groupby(party)["importo"]
        agent = party.strip("id_")
        std = grouped.std().rename(agent + "_std")
        skewness = grouped.agg(sample_skewness)
        skewness = skewness.rename(agent + "_skewness")
        kurtosis = grouped.agg(sample_kurtosis)
        kurtosis = kurtosis.rename(agent + "_kurtosis")
        moments = pd.concat([std, skewness, kurtosis], axis=1)
        df = df.join(moments, on=party)
    return df


def feature_extraction(df):
    # median public expendire by stazione appaltante and year
    df = extract_median_yearly_revenue(df, "id_pa")
    # median revenue by business entity and year
    df = extract_median_yearly_revenue(df, "id_be")

    # median rev/expenditure contract by year
    # df = extract_median_contract(df, "id_pa")
    # df = extract_median_contract(df, "id_be")

    # # median number of contracts by year
    # df = extract_median_yearly_n_contracts(df, "id_pa")
    # df = extract_median_yearly_n_contracts(df, "id_be")

    # contract duration
    df['duration'] = (df.data_fine - df.data_inizio).dt.days

    # month one-hot encoding
    # df = dummy_month(df)

    # continuous encoding of the day_of_year/month as (sin, cos) couple
    # df = encode_sin_cos(df, "Month")
    # alternative: use repeating radial basis function.
    # see the nvidia developer guide for encoding time related variables for
    # rough introduction to the use radial basis function

    # replace data_inizio to days since base_date to avoid datetime format
    # base_date = df.data_inizio.min()
    # df["daysSinceBaseDate"] = (df.data_inizio - base_date).dt.days
    # df = df.drop(columns=["data_inizio", "data_fine"])

    df = compute_moments(df)

    return df


def remove_yearly_infrequent_entities(df, N=10):
    """
    remove business entities and public authority with a number of issued
    contracts lower than N in at least one year; if one year they won less than
    N contracts, then they are removed. 
    """
    for agent in ["id_be", "id_pa"]:
        col_name = "min_nlots_" + agent
        min_nlots = df.groupby([agent, df.data_inizio.dt.year]).size()
        min_nlots = min_nlots.unstack().min(axis=1).rename(col_name)
        df = df.join(min_nlots, on=agent)
        df = df[df[col_name] > N]
        df = df.drop(columns=[col_name])
    return df


def remove_infrequent_entities(df: pd.DataFrame, N=10) -> pd.DataFrame:
    """
    Remove contract lots belonging to entities with a number contracts lower 
    than N over the three years

    """
    for entity in ["id_be", "id_pa"]:
        column_name = entity + "_n_lots"
        n_lots = df.groupby(entity).size().rename(column_name)
        df = df.join(n_lots, on=entity)
        df = df[df[column_name] > N]
        df = df.drop(columns=[column_name])
    return df


# def main_award_procedures(df, proc_list=[1, 4, 23, 26]) -> pd.DataFrame:
#     mask = df["id_scelta_contraente"] == proc_list[0]
#     for proc in proc_list[:][1:]:
#         mask += df["id_scelta_contraente"] == proc
#     return df[mask]


def main():
    df = load_dataset()
    df = split_sum_totals(df)
    # remove the resulting duplicates
    df = df[~df.duplicated()]  # shape 755660, 16
    print(f"df after removing duplicates {df.shape}")
    # df = remove_yearly_infrequent_entities(df, N=10)
    df = remove_infrequent_entities(df, N=10)  # shape 599177, 16
    # df = remove_infrequent_entities(df, N=20)
    print(f"df after removing infrequent entities {df.shape}")
    df = feature_extraction(df)
    df = df.rename(columns={
        "importo": "amount",
        "oggetto": "object",
        "data_inizio": "start_date",
        "id_scelta_contraente": "id_award_procedure"
    })
    return df


if __name__ == "__main__":
    df = main()
    fname = path.join(OUTDIR, "contracts.csv")
    df.to_csv(fname, index_label="index", quoting=csv.QUOTE_NONNUMERIC)
