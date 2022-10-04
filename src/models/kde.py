import time
from sklearn.neighbors import KernelDensity, KDTree
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from scipy.stats import boxcox
import numpy as np
import pandas as pd
import json


class KernelDensityEstimator:

    def __init__(self) -> None:
        self.model = KernelDensity()
        self.features = ['amount', 'pa_med_ann_expenditure',
                         'be_med_ann_revenue', 'duration']
        self.scaler = RobustScaler(with_centering=False)
        params = {"bandwidth": np.logspace(-2, 1, 4)}
        self.optimizer = GridSearchCV(
            estimator=self.model, param_grid=params, cv=3, verbose=1
        )

    def preprocess(self, dataset: pd.DataFrame) -> np.array:
        for f in self.features:
            assert f in dataset.columns, f"{f} not in input dataset features"
        # trim the unused features
        X = dataset[self.features]

        # preprocessing
        X.duration = X.duration.replace(0, X.duration.median())
        X = self.scaler.fit_transform(X)
        # scale only the real-valued columns
        for i in range(len(self.features)):
            X[:, i], _ = boxcox(X[:, i])
        return X

    def opt_params(self, X: np.array) -> None:
        start = time.time()
        self.optimizer.fit(X)
        self.time_elapsed = time.time() - start
        # print(f"model selection required {elapsed:2f} s")
        # print("best params: {0}".format(
        #     self.optimizer.best_estimator_.bandwidth))
        # questo restituisce tutte le specifiche del migliore modello
        # print(self.optimizer.best_estimator_.__dict__)
        # for p in self.params:
        #     optimal_params[p] = best_estimator[p]
        self.model = self.optimizer.best_estimator_

    def test(self, X: np.array) -> np.array:
        return self.model.score_samples(X)

    def config(self):
        obj = dict()
        obj["model"] = self.model.__class__.__name__
        obj["features"] = self.features
        obj[self.scaler.__class__.__name__] = self.scaler.__dict__
        obj["boxcox"] = "function"
        opt_name = self.optimizer.__class__.__name__
        obj[opt_name] = dict()
        obj[opt_name]["param_grid"] = self.optimizer.__dict__["param_grid"]
        obj[opt_name]["cv"] = self.optimizer.__dict__["cv"]
        obj["elapsed_time"] = self.time_elapsed
        obj["cv_results_"] = self.optimizer.__dict__["cv_results_"]
        obj["best_estimator"] = self.optimizer.best_estimator_.__dict__
        return json.dumps(obj, cls=CustomJSONEncoder, indent=4)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return list(o)
        if isinstance(o, np.int32):
            return int(o)
        if isinstance(o, KDTree):
            return KDTree.__name__
        return super().default(o)
