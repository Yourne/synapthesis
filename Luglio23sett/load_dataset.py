from os import path
import numpy as np
import pandas as pd
import utils

def load_dataset(data_directory, lotti_fn, vincitori_fn, procedura_fn):    
    lotti = pd.read_csv(path.join(data_directory, lotti_fn), index_col="id_lotto")
    vincitori = pd.read_csv(path.join(data_directory, vincitori_fn), index_col="id_lotto")
    tipi_procedura = pd.read_csv(path.join(data_directory, procedura_fn), index_col="id_scelta_contraente")
    
    # convert datatypes
    lotti.data_inizio = pd.to_datetime(lotti.data_inizio, yearfirst=True)
    lotti.data_fine = pd.to_datetime(lotti.data_fine, yearfirst=True)
    lotti.data_inferita = pd.to_datetime(lotti.data_inferita, yearfirst=True)
    
    # replace missing values in col1 with values from col2
    lotti = utils.replace_missing_value(lotti, "importo", "importo_base_asta")
    lotti = utils.replace_missing_value(lotti, "data_inizio", "data_inferita")
    
    # drop table attributes with mostly missing values
    lotti = lotti.drop(columns=["oggetto", "importo_liquidato", "importo_base_asta", "data_inferita", "id_mod_realizz", "cpv_vero"])
    print("columns dropped with mostly missing values:")
    print(str(["oggetto", "importo_liquidato", "importo_base_asta", "data_inferita", "id_mod_realizz", "cpv_vero"]))
    
    # clean the dfs from the remaining missing values
    lotti = lotti.dropna()
    vincitori = vincitori.dropna()
    print("dropped all the rows with at least one missing value")
    
    # cast to int64 cols now w/out np.nan
    lotti.id_scelta_contraente = lotti.id_scelta_contraente.astype('int')
    lotti.id_lsf = lotti.id_lsf.astype('int')
    lotti.cpv = lotti.cpv.astype('int')
    
    # merge the datasets
    df = lotti.merge(vincitori, on="id_lotto", how="inner", suffixes=("_pa", "_be"))
    df = df.join(tipi_procedura, on="id_scelta_contraente", how="left")
    
    return df

def feature_extraction(df):
    # median public expendire by stazione appaltante and year
    df = utils.extract_med_rev_by_year(df, "id_pa")
    # median revenue by business entity and year
    df = utils.extract_med_rev_by_year(df, "id_be")
    
    # median rev/expenditure contract by year
    df = utils.extract_med_contract(df, "id_pa")
    df = utils.extract_med_contract(df, "id_be")
    
    # contract duration
    df['durata'] = df.data_fine - df.data_inizio
    
    # continuous encoding of the day_of_year as (sin, cos) couple
    df = utils.encode_sin_cos(df, "day_of_year")
    
    # median number of contracts by year
    df = utils.extract_med_n_contr_by_year(df, "id_pa")
    df = utils.extract_med_n_contr_by_year(df, "id_be")
    
    
    return df



