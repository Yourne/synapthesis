import pandas as pd
import numpy as np
from os import path

import_directory = "synData6July"
lotti_fn = "export_lotti_veneto_2016_2018_giulio_v2.csv"
vincitori_fn = "export_vincitori_veneto_2016_2018_giulio_v2.csv"
output_directory = "datasets"


def replace_missing_value(df, col, replacement_col):
    mask = df[replacement_col].notna() & df[col].isna()
    df.loc[df[mask].index, col] = df[replacement_col][mask]
    return df


def load_dataset():
    # load dataset
    lotti = pd.read_csv(
        path.join(import_directory, lotti_fn), index_col="id_lotto")
    vincitori = pd.read_csv(
        path.join(import_directory, vincitori_fn), index_col="id_lotto")

    # convert datatypes
    lotti.data_inizio = pd.to_datetime(lotti.data_inizio, yearfirst=True)
    lotti.data_fine = pd.to_datetime(lotti.data_fine, yearfirst=True)
    lotti.data_inferita = pd.to_datetime(lotti.data_inferita, yearfirst=True)

    # replace missing values in col1 with values from col2
    lotti = replace_missing_value(lotti, "importo", "importo_base_asta")
    lotti = replace_missing_value(lotti, "data_inizio", "data_inferita")

    # drop table attributes with mostly missing values
    missingValuesCols = [
        "oggetto", "importo_liquidato", "importo_base_asta", "data_inferita",
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
    return lotti.merge(
        vincitori, on="id_lotto", how="inner", suffixes=("_pa", "_be"))


def extract_med_rev_by_year(df, agent):
    rev_by_year = df.groupby([agent, df.data_inizio.dt.year]).sum().importo
    rev_by_year = rev_by_year.unstack()
    med_yearly_rev = rev_by_year.median(axis=1)
    if agent == "id_pa":
        med_yearly_rev = med_yearly_rev.rename("median_annual_expenditure")
    else:
        med_yearly_rev = med_yearly_rev.rename("median_annual_revenue")
    return df.join(med_yearly_rev, on=agent)


def extract_med_contract(df, agent):
    # dovrebbe essere per anno?
    contr_med_agent = df.groupby(agent).median().importo
    contr_med_agent = contr_med_agent.rename(
        "median_contract_" + agent.strip("id_"))
    return df.join(contr_med_agent, on=agent)


def extract_med_n_contr_by_year(df, agent):
    contr_by_year = df.groupby(
        [agent, df.data_inizio.map(lambda x: x.year)]).size()
    contr_by_year = contr_by_year.unstack()
    med_yearly_n_contr = contr_by_year.median(axis=1)
    med_yearly_n_contr = med_yearly_n_contr.rename(
        "med_yearly_n_contr_" + agent.strip("id_"))
    return df.join(med_yearly_n_contr, on=agent)


def encode_sin_cos(df, period="DayOfYear"):
    if period == "month":
        x = df.data_inizio.dt.month
        period_items = 12
    x = df.data_inizio.dt.day_of_year
    period_items = 365
    df["sin" + period] = np.sin(x / period_items * 2 * np.pi)
    df["cos" + period] = np.cos(x / period_items * 2 * np.pi)
    return df


def feature_extraction(df):
    # median public expendire by stazione appaltante and year
    df = extract_med_rev_by_year(df, "id_pa")
    # median revenue by business entity and year
    df = extract_med_rev_by_year(df, "id_be")

    # median rev/expenditure contract by year
    df = extract_med_contract(df, "id_pa")
    df = extract_med_contract(df, "id_be")

    # median number of contracts by year
    df = extract_med_n_contr_by_year(df, "id_pa")
    df = extract_med_n_contr_by_year(df, "id_be")

    # contract duration
    df['duration'] = (df.data_fine - df.data_inizio).dt.days
    df = df.drop(columns=["data_fine"])

    # continuous encoding of the day_of_year as (sin, cos) couple
    df = encode_sin_cos(df, "DayOfYear")
    # alternative: use repeating radial basis function.
    # see the nvidia developer guide for encoding time related variables for
    # rough introduction to the use radial basis function

    # replace data_inizio to days since base_date to avoid datetime format
    base_date = df.data_inizio.min()
    df["daysSinceBaseDate"] = (df.data_inizio - base_date).dt.days
    df = df.drop(columns=["data_inizio"])

    return df


def remove_obvious_outliers(df):
    # 1. contracts having a value higher than the median annual revenue of the
    # business entity winning the bid and of the median expenditure of the
    # public commissioning body (stazione appaltante). Both the business entity
    # and the commissioning body must have a median annual number of contracts
    # higher or equal than five.
    min_year_contr_th = 5
    rev_exp_mask = (df.importo > df.median_annual_expenditure) & \
        (df.importo > df.median_annual_revenue)
    min_year_contr_mask = (df.med_yearly_n_contr_be > min_year_contr_th) & \
        (df.med_yearly_n_contr_pa > min_year_contr_th)
    df = df[~(rev_exp_mask & min_year_contr_mask)]

    # 2. affidamenti diretti having contract duration lasting longer than 10
    # years
    n_years = 10
    years_mask = (df.id_scelta_contraente == 23) & \
        (df.duration > n_years * 365)
    df = df[~years_mask]

    # 3. contracts having a value 25 times higher than the median revenue of
    # business entity and more than 5 contracts (median)
    coef = 25
    coef_mask = (df.importo > coef * df.median_annual_revenue) & \
        (df.med_yearly_n_contr_be > 5)
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


def save_csv_file(df):
    df = df.drop(columns=[
        "id_pa", "id_lsf", "id_be", "med_yearly_n_contr_pa",
        "med_yearly_n_contr_be"])
    df = df.rename(columns={"importo": "sum_total"})
    for cpv, cpv_name in abc_cpv_short_names.items():
        for procedure, procedure_name in abc_procedure_short_names.items():
            mask = (df.cpv == cpv) & (df.id_scelta_contraente == procedure)
            data = df[mask].copy()
            data = data.drop(columns=["cpv", "id_scelta_contraente"])
            file_name = cpv_name + "_" + procedure_name + ".csv"
            data.to_csv(path.join(output_directory, file_name))


if __name__ == "__main__":
    df = load_dataset()
    df = feature_extraction(df)
    df = remove_obvious_outliers(df)
    save_csv_file(df)
