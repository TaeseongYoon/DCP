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

def dcp_qr(Y0, X0, Y1, X1, Y_test, X_test, taus, alpha_sig):
    
    # STEP1. T1의 데이터를 이용해서 각 tau (quantile)에 대해서 beta_qr (Quantile regression의 parameter estimate)
    beta_qr = np.empty((X0.shape[1] + 1, len(taus)))
    for t, tau in enumerate(taus):
        model = sm.QuantReg(Y0, sm.add_constant(X0))
        res = model.fit(q=tau)
        beta_qr[:, t] = res.params

    # STEP2. T2의 데이터를 이용해서 rank, conformity socre 계산 
    tQ_yx = sm.add_constant(X1) @ beta_qr           
    Q_yx = np.sort(tQ_yx, axis=1)
    u_hat = np.mean(Q_yx <= Y1[:, None], axis=1)
    cs = np.abs(u_hat - 0.5)

    tQ_test = sm.add_constant(X_test) @ beta_qr
    Q_test = np.sort(tQ_test, axis=1)
    u_test = np.mean(Q_test <= Y_test[:, None], axis=1)
    cs_test = np.abs(u_test - 0.5)

    # STEP4. Q_tau2_hat (V_t_hat의 empirical quantile 계산)
    k = int(np.ceil((1 - alpha_sig) * (1 + len(Y1))))
    threshold = np.sort(cs)[k]

    ci_grid = np.abs(taus - 0.5)
    cov_qr = cs_test <= threshold              # Conditoinal coveverage

    # Prediction Interval 구하기 
    
    lb = []
    ub = []
    for i in range(len(Y_test)):
        ci = Q_test[i, ci_grid <= threshold]
        ub.append(np.max(ci))
        lb.append(np.min(ci))
        
    UB = np.array(ub)
    LB = np.array(lb)

    leng_qr = UB - LB                           # Prediction Interval의 legnth
    leng_qr[leng_qr == -np.inf] = np.nan

    return UB, LB, cov_qr, leng_qr


def dcp_opt(Y0, X0, Y1, X1, Y_test, X_test, taus, alpha_sig):
    XXX = np.vstack((X1, X_test))
    YYY = np.concatenate((Y1, Y_test))

    beta_qr = np.empty((X0.shape[1] + 1, len(taus)))
    for t, tau in enumerate(taus):
        model = sm.QuantReg(Y0, sm.add_constant(X0))
        res = model.fit(q=tau)
        beta_qr[:, t] = res.params

    tQ_yx = sm.add_constant(XXX) @ beta_qr
    Q_yx = np.sort(tQ_yx, axis=1)
    u_hat = np.mean(Q_yx <= YYY[:, None], axis=1)

    bhat = np.empty(len(YYY))
    b_grid = taus[taus <= alpha_sig]
    for t in range(len(YYY)):
        leng = np.empty(len(b_grid))
        for b, b_val in enumerate(b_grid):
            Q_yx_u = np.interp(b_val + 1 - alpha_sig, taus, Q_yx[t, :])
            leng[b] = Q_yx_u - Q_yx[t, b]
        bhat[t] = b_grid[np.argmin(leng)]

    ind_test = np.arange(len(Y1), len(YYY))
    cs_opt = np.abs(u_hat - bhat - (1 - alpha_sig) / 2)

    k = int(np.ceil((1 - alpha_sig) * (1 + len(Y1))))
    threshold = np.sort(cs_opt[:len(Y1)])[k]

    cov_opt = cs_opt[ind_test] <= threshold

    leng_opt = []
    UB = []
    LB = []
    
    for t in ind_test:
        ci_grid = np.abs(taus - bhat[t] - (1 - alpha_sig) / 2)
        ci = Q_yx[t, ci_grid <= threshold]
        ub = np.max(ci)
        lb = np.min(ci)
        
        leng_opt.append(ub - lb)
        UB.append(ub)
        LB.append(lb)

    leng_opt = np.array(leng_opt)
    leng_opt[leng_opt == -np.inf] = np.nan

    return UB, LB, cov_opt, leng_opt


def dcp_dr(Y0, X0, Y1, X1, Y_test, X_test, ys, taus, alpha_sig):
    beta_dr = np.empty((X0.shape[1] + 1, len(ys)))
    for y, y_val in enumerate(ys):
        model = sm.Logit(Y0 <= y_val, sm.add_constant(X0))
        res = model.fit(disp=0)
        beta_dr[:, y] = res.params

    tF_yx = logistic.cdf(sm.add_constant(X1) @ beta_dr)
    F_yx = np.sort(tF_yx, axis=1)

    cs = np.empty(len(Y1))
    for t in range(len(Y1)):
        u_hat = np.interp(Y1[t], ys, F_yx[t, :])
        cs[t] = np.abs(u_hat - 0.5)

    tF_test = logistic.cdf(sm.add_constant(X_test) @ beta_dr)
    F_test = np.sort(tF_test, axis=1)

    cs_test = np.empty(len(Y_test))
    for t in range(len(Y_test)):
        u_hat = np.interp(Y_test[t], ys, F_test[t, :])
        cs_test[t] = np.abs(u_hat - 0.5)

    k = int(np.ceil((1 - alpha_sig) * (1 + len(Y1))))
    threshold = np.sort(cs)[k]

    cov_dr = cs_test <= threshold

    lb = np.empty(len(Y_test))
    ub = np.empty(len(Y_test))
    for i in range(len(Y_test)):
        ci = ys[np.abs(F_test[i, :] - 0.5) <= threshold]
        ub[i] = np.max(ci)
        lb[i] = np.min(ci)
        
    UB = np.array(UB)
    LB = np.array(LB)

    leng_dr = UB - LB
    leng_dr[leng_dr == -np.inf] = np.nan

    return UB, LB, cov_dr, leng_dr