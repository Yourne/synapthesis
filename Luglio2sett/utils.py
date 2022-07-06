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

