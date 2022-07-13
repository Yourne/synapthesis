import numpy as np

data_directory = "/Users/nepal/Documents/synapthesis/synData6July"
lotti_fn = "export_lotti_veneto_2016_2018_giulio_v2.csv"
vincitori_fn = "export_vincitori_veneto_2016_2018_giulio_v2.csv"
procedura_fn = "/Users/nepal/Documents/synapthesis/tipi_procedure.txt"


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
    rev_by_year = df.groupby([agent, df.data_inizio.dt.year]).sum().importo
    rev_by_year = rev_by_year.unstack()
    med_yearly_rev = rev_by_year.median(axis=1)
    if agent == "id_pa":
        med_yearly_rev = med_yearly_rev.rename("median_expenditure_pa")
    else:
        med_yearly_rev = med_yearly_rev.rename("median_revenue_be")
    return df.merge(med_yearly_rev, on=agent, how="left")


def extract_med_contract(df, agent):
    # dovrebbe essere per anno?
    contr_med_agent = df.groupby(agent).median().importo
    contr_med_agent = contr_med_agent.rename("contr_med_" + agent.strip("id_"))
    return df.merge(contr_med_agent, on=agent, how="left")

def extract_med_n_contr_by_year(df, agent):
    contr_by_year = df.groupby([agent, df.data_inizio.map(lambda x: x.year)]).size()
    contr_by_year = contr_by_year.unstack()
    med_yearly_n_contr = contr_by_year.median(axis=1)
    med_yearly_n_contr = med_yearly_n_contr.rename("med_yearly_n_contr_" + agent.strip("id_"))
    return df.merge(med_yearly_n_contr, on=agent, how="left")
                              
def encode_sin_cos(df, period="year"):
    if period == "month":
        x = df.data_inizio.dt.month
        period = 12
    x = df.data_inizio.dt.day_of_year
    period_days = 365
    df[period + "_sin"] = np.sin(x / period_days * 2 * np.pi)
    df[period + "_cos"] = np.cos(x / period_days * 2 * np.pi)
    return df 

def plot_abc_items(df, category, ax, percentage):
    aggregate = df.groupby(category).importo.sum().sort_values(ascending=False)
    values = (aggregate / aggregate.sum()).values
    labels = aggregate.index
    idx = list(range(len(values)))
    ax.axhline(percentage, c="red", ls="dotted", alpha=.7)
    ax.text(x=idx[-10], y=(percentage -.05), s=f"{percentage} sum amount")
    ax.bar(idx, values)
    ax.plot(np.cumsum(values))
    ax.set_xticks(idx, labels)
    ax.set_xlabel(category)
    ax.set_ylabel("revenue percentage")
    count_aggregate = df.groupby(category).size()
    items = labels[np.cumsum(values)<=percentage].values
    print(f"category within {percentage} - ROW COUNT")
    print(count_aggregate.loc[items])
    return items

