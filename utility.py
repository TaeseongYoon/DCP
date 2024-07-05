import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from datetime import datetime

from scipy.stats import logistic
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

def aux_uneven(Y, X):
    T01 = len(Y)
    T01even = (T01 // 2) * 2

    Yeven = Y[-T01even:]
    Xeven = X[-T01even:]

    ind0 = np.arange(T01even // 2)
    ind1 = np.arange(T01even // 2, T01even)

    Y0 = Yeven[ind0]
    X0 = Xeven[ind0, :]
    Y1 = Yeven[ind1]
    X1 = Xeven[ind1, :]

    return Y0, X0, Y1, X1

# Graphs based on quantile bins
def binning(X, res_mat, num_seg):
    cond = np.empty((num_seg, res_mat.shape[1]))
    xs = np.empty(num_seg)
    quantiles = np.linspace(0, 1, num_seg + 1)
    for i in range(1, num_seg + 1):
        q1 = np.quantile(X, quantiles[i])
        q0 = np.quantile(X, quantiles[i - 1])
        ind = (X <= q1) & (X > q0)

        cond[i - 1, :] = np.nanmean(res_mat[ind.squeeze()], axis=0)
        xs[i - 1] = (q1 + q0) / 2  # midpoint
    return {'xs': xs, 'cond': cond}