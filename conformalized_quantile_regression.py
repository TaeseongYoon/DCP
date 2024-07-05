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

def cqr(Y0, X0, Y1, X1, Y_test, X_test, alpha_sig):
    model_lo = sm.QuantReg(Y0, sm.add_constant(X0))
    model_hi = sm.QuantReg(Y0, sm.add_constant(X0))
    model_50 = sm.QuantReg(Y0, sm.add_constant(X0))
    
    beta_lo = model_lo.fit(q=alpha_sig / 2).params
    beta_hi = model_hi.fit(q=1 - alpha_sig / 2).params
    beta_50 = model_50.fit(q=0.5).params

    tq_lo = sm.add_constant(X1) @ beta_lo
    tq_hi = sm.add_constant(X1) @ beta_hi
    tq_50 = sm.add_constant(X1) @ beta_50

    qsr = np.sort(np.vstack([tq_lo, tq_50, tq_hi]).T, axis=1)

    q_lo = qsr[:, 0]
    q_50 = qsr[:, 1]
    q_hi = qsr[:, 2]

    Eo_vec = np.maximum(q_lo - Y1, Y1 - q_hi)
    Em_vec = np.maximum((q_lo - Y1) / (q_50 - q_lo), (Y1 - q_hi) / (q_hi - q_50))
    Er_vec = np.maximum((q_lo - Y1) / (q_hi - q_lo), (Y1 - q_hi) / (q_hi - q_lo))

    k = int(np.ceil((1 - alpha_sig) * (1 + len(Y1))))
    Q_Eo = np.sort(Eo_vec)[k]
    Q_Em = np.sort(Em_vec)[k]
    Q_Er = np.sort(Er_vec)[k]

    tq_test_lo = sm.add_constant(X_test) @ beta_lo
    tq_test_50 = sm.add_constant(X_test) @ beta_50
    tq_test_hi = sm.add_constant(X_test) @ beta_hi

    qs_test = np.sort(np.vstack([tq_test_lo, tq_test_50, tq_test_hi]).T, axis=1)

    q_test_lo = qs_test[:, 0]
    q_test_50 = qs_test[:, 1]
    q_test_hi = qs_test[:, 2]

    LB_o = q_test_lo - Q_Eo
    UB_o = q_test_hi + Q_Eo
    LB_m = q_test_lo - Q_Em * (q_test_50 - q_test_lo)
    UB_m = q_test_hi + Q_Em * (q_test_hi - q_test_50)
    LB_r = q_test_lo - Q_Er * (q_test_hi - q_test_lo)
    UB_r = q_test_hi + Q_Er * (q_test_hi - q_test_lo)

    cov_o = (Y_test <= UB_o) & (Y_test >= LB_o)
    cov_m = (Y_test <= UB_m) & (Y_test >= LB_m)
    cov_r = (Y_test <= UB_r) & (Y_test >= LB_r)

    leng_o = UB_o - LB_o
    leng_m = UB_m - LB_m
    leng_r = UB_r - LB_r
    
    return UB_o, LB_o, UB_m, LB_m, UB_r, LB_r, cov_o, cov_m, cov_r, leng_o, leng_m, leng_r