from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    data_directory = "/Users/nepal/Documents/synapthesis/synData6July"
    lotti_fn = "export_lotti_veneto_2016_2018_giulio_v2.csv"
    vincitori_fn = "export_vincitori_veneto_2016_2018_giulio_v2.csv"

    lotti = pd.read_csv(path.join(data_directory, lotti_fn))
    vincitori = pd.read_csv(path.join(data_directory, vincitori_fn))

