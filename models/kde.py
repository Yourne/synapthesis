from sklearn.neighbors import KernelDensity
# from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from scipy.stats import boxcox
import pandas as pd
from os import path

if __name__ == "__main__":
    directory = "datasets"
    dataset_name = "appMed_aperta"
    model = "kde"
    dataset = pd.read_csv(
        path.join(directory, dataset_name + ".csv"), index_col="idx")

    # remove id_be, id_pa, id_lsf
    X = dataset.drop(columns=["id_lotto", "id_pa", "id_be", "id_lsf"])
    # remove time related features
    X = X.drop(columns=["sinDayOfYear", "cosDayOfYear", "daysSinceBaseDate"])
    # remove median contract pa, median contract be, "pa_med_ann_n_contr",
    # "be_med_ann_n_contr" as they are computed on whole dataset, not only the
    # CPV and award procedure
    X = X.drop(columns=['pa_med_ann_contr', 'be_med_ann_contr',
                        'pa_med_ann_n_contr', 'be_med_ann_n_contr'])

    print(X.columns)

    # preprocessing
    col_names = X.columns
    X.duration = X.duration.replace(0, X.duration.median())
    scaler = RobustScaler(with_centering=False)
    X = scaler.fit_transform(X)
    # scale only the real-valued columns
    for i in range(0, 2):
        X[:, i], _ = boxcox(X[:, i])

    # # optimize the bandwidth
    # params = {"bandwidth": np.logspace(-1, 1, 20)}
    # print(params)
    # grid = GridSearchCV(KernelDensity(), params, verbose=1)
    # grid.fit(X)

    # print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    # # best bandwith: 0.6951927961775606

    # # use the best estimator to compute the kernel density estimate
    # kde = grid.best_estimator_

    # already optimized bandwidth
    bandwidth = 0.26366508987303583
    kde = KernelDensity(bandwidth=1, kernel="gaussian")
    kde.fit(X)

    # likelihoods
    scores = kde.score_samples(X)

    # plot and save results
    # utils.plot(X, scores, model, dataset_name)
