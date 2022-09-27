import pandas as pd
import numpy as np
from os import path

OUTDIR = "data"
INPUTDIR = "synData6July"
LOTTI_FNAME = "export_lotti_veneto_2016_2018_giulio_v2.csv"
VINCITORI_FNAME = "export_vincitori_veneto_2016_2018_giulio_v2.csv"


def replace_missing_value(df, col, replacement_col):
    mask = df[replacement_col].notna() & df[col].isna()
    df.loc[df[mask].index, col] = df[replacement_col][mask]
    return df


def load_dataset(INPUTDIR, LOTTI_FNAME, VINCITORI_FNAME):
    lotti = pd.read_csv(path.join(INPUTDIR, LOTTI_FNAME))
    vincitori = pd.read_csv(path.join(INPUTDIR, VINCITORI_FNAME))

    # convert datatypes
    lotti.data_inizio = pd.to_datetime(lotti.data_inizio, yearfirst=True)
    lotti.data_fine = pd.to_datetime(lotti.data_fine, yearfirst=True)
    lotti.data_inferita = pd.to_datetime(lotti.data_inferita, yearfirst=True)

    # replace missing values in col1 with values from col2
    lotti = replace_missing_value(lotti, "importo", "importo_base_asta")
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

    # print("dropped all the rows with at least one missing value")
    # cast to int64 cols now w/out np.nan
    lotti.id_scelta_contraente = lotti.id_scelta_contraente.astype('int')
    lotti.id_lsf = lotti.id_lsf.astype('int')
    lotti.cpv = lotti.cpv.astype('int')
    # drop columns that lead to sparse matrices
    # following tests: try models that handle sparse datasets
    lotti = lotti.drop(
        columns=["id_forma_giuridica", "uber_forma_giuridica"])
    vincitori = vincitori.drop(
        columns=["id_forma_giuridica", "uber_forma_giuridica"])

    # merge the datasets
    return lotti.merge(vincitori, on="id_lotto")


def split_sum_totals(df):
    # remove duplicates
    df = df[~df.duplicated()]
    # save records having more than one winner in a temporary variable
    temp = df[df["id_lotto"].duplicated(keep=False)]
    # equally split the lot sum among all the winners
    # (uniform distr. is uninformative prior)
    temp = temp.join(temp.groupby("id_lotto").size().rename("n_winners"),
                     on="id_lotto")
    temp = temp["importo"] / temp["n_winners"]
    # modifico la copia o l'originale?
    # l'originale secondo quello che so
    df.loc[temp.index, "importo"] = temp
    return df


def extract_median_yearly_revenue(df, agent):
    rev_by_year = df.groupby([agent, df.data_inizio.dt.year]).sum().importo
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


def feature_extraction(df):
    # median public expendire by stazione appaltante and year
    df = extract_median_yearly_revenue(df, "id_pa")
    # median revenue by business entity and year
    df = extract_median_yearly_revenue(df, "id_be")

    # median rev/expenditure contract by year
    df = extract_median_contract(df, "id_pa")
    df = extract_median_contract(df, "id_be")

    # median number of contracts by year
    df = extract_median_yearly_n_contracts(df, "id_pa")
    df = extract_median_yearly_n_contracts(df, "id_be")

    # contract duration
    df['duration'] = (df.data_fine - df.data_inizio).dt.days

    # month one-hot encoding
    df = dummy_month(df)

    # continuous encoding of the day_of_year/month as (sin, cos) couple
    df = encode_sin_cos(df, "Month")
    # alternative: use repeating radial basis function.
    # see the nvidia developer guide for encoding time related variables for
    # rough introduction to the use radial basis function

    # replace data_inizio to days since base_date to avoid datetime format
    base_date = df.data_inizio.min()
    df["daysSinceBaseDate"] = (df.data_inizio - base_date).dt.days
    # df = df.drop(columns=["data_inizio", "data_fine"])

    return df


def remove_obvious_outliers(df):
    # 1. contracts having a value higher than the median annual revenue of the
    # business entity winning the bid and of the median expenditure of the
    # public commissioning body (stazione appaltante). Both the business entity
    # and the commissioning body must have a median annual number of contracts
    # higher or equal than five.

    min_yearly_n_contr = 5
    revenue_mask = (df.importo > df.pa_med_ann_expenditure) & \
        (df.importo > df.be_med_ann_revenue)
    min_year_contr_mask = (df.be_med_ann_n_contr > min_yearly_n_contr) & \
        (df.pa_med_ann_n_contr > min_yearly_n_contr)
    df = df[~(revenue_mask & min_year_contr_mask)]

    # 2. affidamenti diretti having contract duration lasting longer than 10
    # years
    n_years = 10
    years_mask = (df.id_scelta_contraente == 23) & \
        (df.duration > n_years * 365)
    df = df[~years_mask]

    # 3. contracts having a value 25 times higher than the median revenue of
    # business entity and more than 5 contracts (median)
    coef = 25
    coef_mask = (df.importo > coef * df.be_med_ann_revenue) & \
        (df.be_med_ann_n_contr > 5)
    df = df[~coef_mask]
    return df


abc_procedure_short_names = {
    1: "aperta",
    26: "adesione",
    4: "negoziata",
    23: "affidamento"
}

abc_cpv_short_names = {
    33: "appMed",
    45: "lavori",
    85: "servSani",
    79: "servImpr"
}


def save_abc_only(df):
    for cpv, cpv_name in abc_cpv_short_names.items():
        for procedure, procedure_name in abc_procedure_short_names.items():
            mask = (df.cpv == cpv) & (df.id_scelta_contraente == procedure)
            data = df[mask].copy()
            data = data.drop(columns=["id_scelta_contraente"])
            file_name = cpv_name + "_" + procedure_name + ".csv"
            data.to_csv(path.join(OUTDIR, file_name),
                        index_label="idx")


def save_award_procedure(df, procedure_id, split="train"):
    data = df[df["id_scelta_contraente"] == procedure_id]
    data = data.drop(columns=["id_scelta_contraente", "cpv"])
    fname = path.join(OUTDIR, abc_procedure_short_names[procedure_id] + "_" +
                      split)
    data.to_csv(fname, index_label="idx")


if __name__ == "__main__":
    df = load_dataset(INPUTDIR, LOTTI_FNAME, VINCITORI_FNAME)
    df = split_sum_totals(df)
    df = feature_extraction(df)
    df = remove_obvious_outliers(df)
    df = df.rename(columns={"importo": "amount"})
    # save_abc_only(df)

    # train test split by year
    df["year"] = df["data_inizio"].dt.year
    df_tr = df[(df["year"] == 2016) | (df["year"] == 2017)]
    df_te = df[df["year"] == 2018]
    save_award_procedure(df_tr, "train")
    save_award_procedure(df_tr, "test")