def scatterplot_durata(df, id_scelta_contraente, ax, thr=5):
    th = 365 * thr
    if th:
        df = df[df.durata.dt.days < th]
    ax.scatter(x=df.durata.dt.days, y=df.importo, s=10,
               alpha=.3, c=color_procedure[id_scelta_contraente], 
               label=abc_procedure_names[id_scelta_contraente])
    for i in range(thr):
        ax.axvline(i*365, ls="dotted", c="black", alpha=.3)
    ax.set_xticks(np.arange(0, thr, 1) * 365, [str(el) for el in np.arange(0, 5, 1)])
    ax.set_yscale("log")
    ax.set_xlabel("years")
    ax.set_ylabel("amount")
    ax.legend()
    
def scatterplot_revenue(df, id_scelta_contraente, ax):
    ax.scatter(x=df.median_revenue_be, y=df.importo, s=10,
               alpha=.3, c=color_procedure[id_scelta_contraente], 
               label=abc_procedure_names[id_scelta_contraente])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("amount")
    ax.legend()
    
def plot_quaternion(df, cpv, func):
    fig, ax = plt.subplots(2, 2, figsize=(6*2.5, 4*3))
    for i, id_scelta_contraente in enumerate(abc_procedure_names):
        table = df[(df.cpv == cpv) & (df.id_scelta_contraente == id_scelta_contraente)]
        func(table, id_scelta_contraente, ax[i//2, i%2])
        fig.suptitle(f"{cpv} - {abc_cpv_names[cpv]}")
    plt.tight_layout()