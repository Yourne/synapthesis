import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from scipy.stats import boxcox
from sklearn.preprocessing import RobustScaler
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer
# import time
import json


def outlier_proportion_score(preds, n_outliers):
    # estimated correct proportion of outliers
    c = n_outliers / len(preds)
    # number of predicted outliers
    x = (sum(preds) - len(preds)) / 2
    return -x**2 - c


class OneClassSVMEstimator:
    def __init__(self) -> None:
        self.model = OneClassSVM()
        self.features = ['amount', 'pa_med_ann_expenditure',
                         'be_med_ann_revenue', 'duration']
        self.scaler = RobustScaler(with_centering=False)
        self.n_outliers = None
        self.optimizer = None

    def preprocess(self, dataset: pd.DataFrame) -> np.array:
        # record the number of outliers
        self.n_outliers = sum(dataset["outlier"])

        # check the required features
        for f in self.features:
            assert f in dataset.columns, f"{f} not in input dataset features"

        # trim unused features
        X = dataset[self.features]

        # prerpocessing
        X.duration = X.duration.replace(0, X.duration.median())
        X = self.scaler.fit_transform(X)
        # scale only the real-valued columns
        for i in range(len(self.features)):
            X[:, i], _ = boxcox(X[:, i])
        return X

    def opt_params(self, X: np.array) -> None:
        nu = self.n_outliers / X.shape[0]
        
        # params = {"nu": [nu - i for i in [10e-4, 0, -10e-3]]}
        # print(params)
        # # GridSearchCV requires a scoring function.
        # # The proportion of outliers, assuming to know it.
        # score = make_scorer(outlier_proportion_score)
        # self.optimizer = GridSearchCV(
        #     estimator=self.model, param_grid=params, scoring=score
        # )
        # start = time.time()
        # self.optimizer.fit(X)
        # self.time_elapsed = time.time() - start
        # self.model = self.optimizer.best_estimator_

        self.model = OneClassSVM(nu=nu)
        self.model.fit(X)

    def test(self, X: np.array) -> np.array:
        return self.model.predict(X)

    def config(self) -> str:
        obj = dict()
        obj["model"] = self.model.__class__.__name__
        obj["features"] = self.features
        obj[self.scaler.__class__.__name__] = self.scaler.__dict__
        obj["boxcox"] = "function"
        obj["best_estimator"] = self.model.__dict__
        # opt_name = self.optimizer.__class__.__name__
        # obj[opt_name] = dict()
        # obj[opt_name]["param_grid"] = self.optimizer.__dict__["param_grid"]
        # obj[opt_name]["cv"] = self.optimizer.__dict__["cv"]
        # obj["elapsed_time"] = self.time_elapsed
        # obj["cv_results_"] = self.optimizer.__dict__["cv_results_"]
        # obj["best_estimator"] = self.optimizer.best_estimator_.__dict__
        return json.dumps(obj, indent=4, cls=CustomJSONEncoder)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return list(o)
        if isinstance(o, np.int32):
            return int(o)
        return super().default(o)
