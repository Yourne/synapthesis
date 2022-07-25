import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

data_directory = "/Users/nepal/Documents/synapthesis/synData6July"
lotti_fn = "export_lotti_veneto_2016_2018_giulio_v2.csv"
vincitori_fn = "export_vincitori_veneto_2016_2018_giulio_v2.csv"
procedura_fn = "/Users/nepal/Documents/synapthesis/tipi_procedure.txt"

abc_procedure_names = {
    1: "procedura aperta",
    26: "adesione",
    4: "procedura negoziata",
    23: "affidamento diretto"
}

abc_cpv_names = {
    33: "Apparecchiature mediche",
    45: "Lavori di costruzione",
    85: "Servizi sanitari",
    79: "Servizi per le imprese"
}

color_procedure = dict()
for id_scelta_contraente, color in zip(abc_procedure_names.keys(),
                                       mcolors.TABLEAU_COLORS):
    color_procedure[id_scelta_contraente] = color


def print_df_measures(df):
    print(f"SHAPE: {df.shape}")
    print()
    print("DTYPES")
    print(df.dtypes)
    print()
    print("COMPLETENESS percentages (100 means no NaNs)")
    print(np.sum(df.notna(), axis=0) / df.shape[0] * 100)


def replace_missing_value(df, col, replacement_col):
    temp = sum(df[replacement_col][df[col].isna()].notna()) / df.shape[0]
    print(f"{col} values to substitute with {replacement_col} values \
          percentage: {temp:.4f}%")

    mask = df[replacement_col].notna() & df[col].isna()
    df.loc[df[mask].index, col] = df[replacement_col][mask]
    return df


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
    contr_by_year = df.groupby([agent,
                                df.data_inizio.map(lambda x: x.year)]).size()
    contr_by_year = contr_by_year.unstack()
    med_yearly_n_contr = contr_by_year.median(axis=1)
    med_yearly_n_contr = med_yearly_n_contr.rename(
        "med_yearly_n_contr_" + agent.strip("id_"))
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
    ax.text(x=idx[-10], y=(percentage - .05), s=f"{percentage} sum amount")
    ax.bar(idx, values)
    ax.plot(np.cumsum(values))
    ax.set_xticks(idx, labels)
    ax.set_xlabel(category)
    ax.set_ylabel("revenue percentage")
    ax.set_title(category)
    count_aggregate = df.groupby(category).size()
    items = labels[np.cumsum(values) <= percentage].values
    print(f"category within {percentage} - ROW COUNT")
    print(count_aggregate.loc[items])
    return items


def scatter_quaternion(xcol, ycol, df, cpv, max_years=5):
    """x, y : df column labels"""
    fig, ax = plt.subplots(
        2, 2, figsize=(6*2.5, 4*3), sharex=True, sharey=True)
    fig.suptitle(f"{cpv} - {abc_cpv_names[cpv]}")
    for i, proc in enumerate(abc_procedure_names):
        axx = ax[i // 2, i % 2]
        table = df[(df.cpv == cpv) & (df.id_scelta_contraente == proc)]
        # print(table[["cpv", "id_scelta_contraente"]])

        if xcol == "durata":
            table = table[table.durata.dt.days < 365 * max_years]
            x = table.durata.dt.days
            # years = np.arange(0, max_years, 1)
            # axx.set_xticks(years * 365, [str(el) for el in years])
            
            # for i in range(max_years):
            #     axx.axvline(i*365, ls="dotted", c="black", alpha=.3)
            axx.set_xscale("log")
            axx.set_xlabel("days")
        else:
            x = table[xcol]
            axx.set_xscale("log")
            axx.set_xlabel(xcol)

        axx.set_yscale("log")
        axx.set_ylabel(ycol)

        axx.scatter(x=x, y=table[ycol], s=10, alpha=.3,
                    c=color_procedure[proc], label=abc_procedure_names[proc])
        axx.legend()

    plt.tight_layout()


def setup_axes(xcol, ycol, table, axx, max_years=10):
    if xcol == "durata":
        table = table[table.durata.dt.days < 365 * max_years]
        x = table.durata.dt.days
        years = np.arange(0, max_years, 1)
        axx.set_xticks(years * 365, [str(el) for el in years])
        axx.set_xlabel("years")
        for i in range(max_years):
            axx.axvline(i*365, ls="dotted", c="black", alpha=.3)
    else:
        x = table[xcol]
        axx.set_xscale("log")
        axx.set_xlabel(xcol)
    y = table[ycol]
    axx.set_yscale("log")
    axx.set_ylabel(ycol)

    return x, y, axx


def compare_filtering(df, xcol, ycol, outlier_category,
                      max_years=10, alpha=.3):
    _, ax = plt.subplots(figsize=(6*3, 4*3))
    x, y, ax = setup_axes(xcol, ycol, df, ax, max_years)
    scatter = ax.scatter(x, y, s=10, alpha=alpha, c=df[outlier_category])
    legend = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend)
