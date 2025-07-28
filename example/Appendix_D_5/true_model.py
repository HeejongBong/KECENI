import os, sys
import numpy as np

import scipy.linalg as la
import scipy.stats as stats
from scipy.special import expit

from hyperparams import HyperAlpha

hyper_true = HyperAlpha(0, 0)

def mu(T_N1, X_N2, G_N2):
    n1 = T_N1.shape[-1]; n2 = X_N2.shape[-2]
    
    Z = hyper_true.summary_mu(T_N1, X_N2, G_N2)
    
    return Z @ [2, 2, -1.55, -1.55, -1.55, -1.55, -1.55, -1.55]
    
def pij(T, X_N1, G_N1):
    Z = hyper_true.summary_pi(X_N1, G_N1)
    
    return np.abs(expit(Z @ [0.5, 0.5, 0.5, 0, 0, 0]) - 1 + T)
    
def pi(T_N1, X_N2, G_N2):
    n1 = T_N1.shape[-1]; n2 = X_N2.shape[-2]
    return np.prod([
        pij(T_N1[...,j], X_N2[...,G_N2.N1(j),:], G_N2.sub(G_N2.N1(j))) 
        for j in np.arange(n1)
    ], 0)

def rX(n_sample, N2, G):
    n2 = N2.shape[0]

    X1 = stats.norm.rvs(0, 1, size=(n_sample,n2))
    X2 = stats.norm.rvs(0, 1, size=(n_sample,n2))
    X3 = stats.norm.rvs(0, 1, size=(n_sample,n2))

    return np.stack([X1, X2, X3], -1)