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

def cp_reg(Y0, X0, Y1, X1, Y_test, X_test, alpha_sig):
    beta_reg = sm.OLS(Y0, sm.add_constant(X0)).fit().params
    cs = np.abs(Y1 - sm.add_constant(X1) @ beta_reg)
    k = int(np.ceil((1 - alpha_sig) * (1 + len(Y1))))
    threshold = np.sort(cs)[k]
    
    cov_reg = np.abs(Y_test - sm.add_constant(X_test) @ beta_reg) <= threshold
    leng_reg = np.full(len(Y_test), 2 * threshold)
    
    UB = sm.add_constant(X_test) @ beta_reg +  threshold
    LB = sm.add_constant(X_test) @ beta_reg - threshold
    
    return UB, LB, cov_reg, leng_reg

def cp_loc(Y0, X0, Y1, X1, Y_test, X_test, alpha_sig):
    beta_reg = sm.OLS(Y0, sm.add_constant(X0)).fit().params
    absR0 = np.abs(Y0 - sm.add_constant(X0) @ beta_reg)
    beta_sig = sm.OLS(absR0, sm.add_constant(X0)).fit().params
    sig1 = np.abs(sm.add_constant(X1) @ beta_sig)
    absR1 = np.abs(Y1 - sm.add_constant(X1) @ beta_reg)
    cs = absR1 / sig1
    k = int(np.ceil((1 - alpha_sig) * (1 + len(Y1))))
    threshold = np.sort(cs)[k]
    
    LB = sm.add_constant(X_test) @ beta_reg - threshold * np.abs(sm.add_constant(X_test) @ beta_sig)
    UB = sm.add_constant(X_test) @ beta_reg + threshold * np.abs(sm.add_constant(X_test) @ beta_sig)
    
    cov_loc = (Y_test <= UB) & (Y_test >= LB)
    leng_loc = UB-LB
    
    return UB, LB, cov_loc, leng_loc