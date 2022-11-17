from sklearn.neighbors import LocalOutlierFactor, KDTree
from sklearn.preprocessing import RobustScaler
from scipy import stats
import pandas as pd
import numpy as np
import json


class LocalOutlierFactorEstimator:
    def __init__(self) -> None:
        self.n_neighbors = 20
        self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors)
        self.features = ["amount", "pa_med_ann_expenditure",
                         "be_med_ann_revenue", "duration"]
        self.scaler = RobustScaler(with_centering=False)

    def preprocess(self, dataset: pd.DataFrame) -> np.array:
        for f in self.features:
            assert f in dataset.columns, f"{f} not in input dataset features"

        X = dataset[self.features]
        X = X.replace({"duration": 0}, 1)
        X = self.scaler.fit_transform(X)
        for i in range(len(self.features)):
            X[:, i], _ = stats.boxcox(X[:, i])
        return X

    def opt_params(self, X: np.array) -> None:
        self.model.fit(X)

    def test(self, X: np.array) -> np.array:
        return self.model.fit_predict(X)

    def config(self) -> str:
        obj = dict()
        obj["model"] = self.model.__class__.__name__
        obj["features"] = self.features
        obj[self.scaler.__class__.__name__] = self.scaler.__dict__
        obj["boxcox"] = "function"
        obj["best_estimator"] = self.model.__dict__
        return json.dumps(obj, indent=4, cls=CustomJSONEncoder)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return list(o)
        elif isinstance(o, np.int32):
            return int(o)
        elif isinstance(o, KDTree):
            return (KDTree.__class__.__name__)
        else:
            return super().default(o)
