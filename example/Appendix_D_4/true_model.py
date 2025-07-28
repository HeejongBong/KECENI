import os, sys
import numpy as np

import scipy.linalg as la
import scipy.stats as stats
from scipy.special import expit

def summary_mu(T_N1, X_N2, G_N2):
    n1 = T_N1.shape[-1]; n2 = X_N2.shape[-2]; d = X_N2.shape[-1]
    
    if n1 == 1:
        return np.stack(np.broadcast_arrays(
            T_N1[...,0] - 0.5, 
            np.full(T_N1[...,0].shape, 0), 
            *np.moveaxis(X_N2[...,0,:] - 0.5, -1, 0), 
            *np.moveaxis(np.full(X_N2[...,0,:].shape, 0), -1, 0),
            (X_N2[...,0,0] - 0.5) * (X_N2[...,0,1] - 0.5), 
            (X_N2[...,0,1] - 0.5) * (X_N2[...,0,2] - 0.5), 
            (X_N2[...,0,2] - 0.5) * (X_N2[...,0,0] - 0.5), 
            *np.moveaxis(np.full(X_N2[...,0,:].shape, 0), -1, 0)
        ), -1)
    else:
        return np.stack(np.broadcast_arrays(
            T_N1[...,0] - 0.5, 
            np.mean(T_N1[...,1:n1], -1) - 0.5, 
            *np.moveaxis(X_N2[...,0,:] - 0.5, -1, 0), 
            *np.moveaxis(np.mean(X_N2[...,1:n1,:], -2) - 0.5, -1, 0),
            (X_N2[...,0,0] - 0.5) * (X_N2[...,0,1] - 0.5), 
            (X_N2[...,0,1] - 0.5) * (X_N2[...,0,2] - 0.5), 
            (X_N2[...,0,2] - 0.5) * (X_N2[...,0,0] - 0.5), 
            np.mean((X_N2[...,1:n1,0] - 0.5) * (X_N2[...,1:n1,1] - 0.5), -1),
            np.mean((X_N2[...,1:n1,1] - 0.5) * (X_N2[...,1:n1,2] - 0.5), -1),
            np.mean((X_N2[...,1:n1,2] - 0.5) * (X_N2[...,1:n1,0] - 0.5), -1)
        ), -1)
    
def summary_pi(X_N1, G_N1):
    n1 = X_N1.shape[-2]
    
    if n1 == 1:
        return np.stack(np.broadcast_arrays(
            *np.moveaxis(X_N1[...,0,:], -1, 0), 
            *np.moveaxis(np.full(X_N1[...,0,:].shape, 0), -1, 0),
            (X_N1[...,0,0] - 0.5) * (X_N1[...,0,1] - 0.5), 
            (X_N1[...,0,1] - 0.5) * (X_N1[...,0,2] - 0.5), 
            (X_N1[...,0,2] - 0.5) * (X_N1[...,0,0] - 0.5), 
            *np.moveaxis(np.full(X_N1[...,0,:].shape, 0), -1, 0)
        ), -1)
    else:
        return np.stack(np.broadcast_arrays( 
            *np.moveaxis(X_N1[...,0,:], -1, 0), 
            *np.moveaxis(np.mean(X_N1[...,1:n1,:], -2), -1, 0),
            (X_N1[...,0,0] - 0.5) * (X_N1[...,0,1] - 0.5), 
            (X_N1[...,0,1] - 0.5) * (X_N1[...,0,2] - 0.5), 
            (X_N1[...,0,2] - 0.5) * (X_N1[...,0,0] - 0.5), 
            np.mean((X_N1[...,1:n1,0] - 0.5) * (X_N1[...,1:n1,1] - 0.5), -1),
            np.mean((X_N1[...,1:n1,1] - 0.5) * (X_N1[...,1:n1,2] - 0.5), -1),
            np.mean((X_N1[...,1:n1,2] - 0.5) * (X_N1[...,1:n1,0] - 0.5), -1)
        ), -1)

def mu(T_N1, X_N2, G_N2):
    n1 = T_N1.shape[-1]; n2 = X_N2.shape[-2]
    
    Z_N2 = summary_mu(T_N1, X_N2, G_N2)
    
    # return 5 + Z_N2 @ [1, 0.35, 0.5, 0.58, 0.33, 0.2, 0.2, 0.1]
    # return (Z_N2 @ [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -7, 0, 0])/10 + 0.5 # 12
    return expit(Z_N2 @ [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -7, 0, 0])
    
def pij(T, X_N1, G_N1):
    Z_N1 = summary_pi(X_N1, G_N1)
    
    return np.abs(expit(Z_N1 @ [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0]) - 1 + T)
    
def pi(T_N1, X_N2, G_N2):
    n1 = T_N1.shape[-1]; n2 = X_N2.shape[-2]
    return np.prod([
        pij(T_N1[...,j], X_N2[...,G_N2.N1(j),:], G_N2.sub(G_N2.N1(j))) 
        for j in np.arange(n1)
    ], 0)

def rX(n_sample, N2, G):
    n2 = N2.shape[0]

    # X1 = stats.uniform.rvs(-1, 2, size=(n_sample,n2))
    X1 = stats.bernoulli.rvs(0.5, size=(n_sample,n2))
    X2 = stats.bernoulli.rvs(0.5, size=(n_sample,n2))
    X3 = stats.bernoulli.rvs(0.5, size=(n_sample,n2))

    return np.stack([X1, X2, X3], -1)

def rW(X):
    return np.stack([
        (X[...,0] - 0.5) * (X[...,1] - 0.5),
        (X[...,1] - 0.5) * (X[...,2] - 0.5),
        (X[...,2] - 0.5) * (X[...,0] - 0.5),
    ], -1)