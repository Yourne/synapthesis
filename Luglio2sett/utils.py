import numpy as np

def print_df_measures(df):
    print(f"SHAPE: {df.shape}")
    print()
    print("DTYPES")
    print(df.dtypes)
    print()
    print("COMPLETENESS percentages (100 means no NaNs)")
    print(np.sum(df.notna(), axis=0) / df.shape[0] * 100)
    
def replace_missing_value(df, col, replacement_col):
    print(f"{col} values to substitute: \
          {sum(df[replacement_col][df[col].isna()].notna()) / df.shape[0] :.4f}%")
    mask = df[replacement_col].notna() & df[col].isna()
    df.loc[df[mask].index, col] = df[replacement_col][mask]
    return df

def compute_revenue_year(df, supplier, date_col):
    """ 
    supplier: column name of the supplier (or buyer) in dataframe df
    date_col: column name of the date of the contract
    return: the total revenue for each year for each supplier (or buyer)
    """
    rev_by_year = df.groupby([supplier, df[date_col].map(lambda x: x.year)]).sum()["importo"]
    rev_by_year = rev_by_year.unstack()
    return rev_by_year

def extract_med_rev_by_year(df, agent):
    rev_by_year = df.groupby([agent, df.data_inizio.map(lambda x: x.year)]).sum().importo
    rev_by_year = rev_by_year.unstack()
    med_yearly_rev = rev_by_year.median(axis=1)
    if agent == "id_pa":
        med_yearly_rev = med_yearly_rev.rename("erogato_med_pa")
    else:
        med_yearly_rev = med_yearly_rev.rename("fatt_med_be")
    return df.merge(med_yearly_rev, on=agent, how="left")


def extract_med_contract(df, agent):
    contr_med_agent = df.groupby(agent).median().importo
    contr_med_agent = contr_med_agent.rename("contr_med_" + agent)
    return df.merge(contr_med_agent, on=agent, how="left")


