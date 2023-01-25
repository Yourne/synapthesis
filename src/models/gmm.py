from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from scipy import stats
import pandas as pd
import numpy as np
import json


class GaussianMixtureEstimator:
    def __init__(self) -> None:
        # best model hyper params
        self.model = GaussianMixture(n_components=50, covariance_type="diag")
        self.features = ["be_amount", "pa_amount",
                         "be_duration", "pa_duration",
                         "be_med_ann_revenue", "pa_med_ann_expenditure"
                         "be_amount_std", "pa_amount_std"
                         ]
        self.scaler = RobustScaler(with_centering=False)
        self.bic = list()
        self.n_components_range = range(1, 60)

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
        lowest_bic = np.infty
        cv_types = ["spherical", "tied", "diag", "full"]
        for cv_type in cv_types:
            for n_components in self.n_components_range:
                # Fit a Gaussian mixture with EM
                gmm = GaussianMixture(
                    n_components=n_components, covariance_type=cv_type
                )
                gmm.fit(X)
                self.bic.append(gmm.bic(X))
                if self.bic[-1] < lowest_bic:
                    lowest_bic = self.bic[-1]
                    # best_gmm = gmm
                    self.model = gmm

    def fit(self, X: np.array):
        self.model.fit(X)

    def test(self, X: np.array) -> np.array:
        return self.model.score_samples(X)

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
        else:
            return super().default(o)
