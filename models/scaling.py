import pandas as pd
# import numpy as np
from os import path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from scipy.stats import boxcox

if __name__ == "__main__":
    directory = "datasets"
    file_name = "appMed_adesione"

    X = pd.read_csv(
        path.join(directory, file_name), index_col="id_lotto")
    col_names = X.columns

    # substitute 0 duration contracts with the mean
    X.duration = X.duration.replace(0, X.duration.median())

    scaler = RobustScaler(with_centering=False)
    X = scaler.fit_transform(X)
    for i in range(0, 6):
        X[:, i], _ = boxcox(X[:, i])
        ax = plt.axes()
        ax.set_title(col_names[i])
        sns.histplot(X[:, i], ax=ax)
        plt.show()
