import numpy as np
import scipy.linalg as la

def summary_mu(T_N1, X_N2, G_N2):
    n1 = T_N1.shape[-1]; n2 = X_N2.shape[-2]; d = X_N2.shape[-1]
    
    if n1 == 1:
        return np.stack(np.broadcast_arrays(
            T_N1[...,0] - 0.5, 
            np.full(T_N1[...,0].shape, 0), 
            *np.moveaxis(X_N2[...,0,:], -1, 0), 
            *np.moveaxis(np.full(X_N2[...,0,:].shape, 0), -1, 0)
        ), -1)
    else:
        return np.stack(np.broadcast_arrays(
            T_N1[...,0] - 0.5, 
            np.mean(T_N1[...,1:n1], -1) - 0.5, 
            *np.moveaxis(X_N2[...,0,:], -1, 0), 
            *np.moveaxis(np.mean(X_N2[...,1:n1,:], -2), -1, 0)
        ), -1)
    
def summary_pi(X_N1, G_N1):
    n1 = X_N1.shape[-2]; d = X_N1.shape[-1]
        
    if n1 == 1:
        return np.stack(np.broadcast_arrays(
            *np.moveaxis(X_N1[...,0,:], -1, 0), 
            *np.moveaxis(np.full(X_N1[...,0,:].shape, 0), -1, 0)
        ), -1)
    else:
        return np.stack(np.broadcast_arrays(
            *np.moveaxis(X_N1[...,0,:], -1, 0), 
            *np.moveaxis(np.mean(X_N1[...,1:n1,:], -2), -1, 0)
        ), -1)
    
def summary(T_N1, G_N2):
    n1 = T_N1.shape[-1]
    
    if n1 == 1:
        return np.stack(np.broadcast_arrays(
            T_N1[...,0] - 0.5, 
            np.full(T_N1[...,0].shape, 0)
        ), -1)
    else:
        return np.stack(np.broadcast_arrays(
            T_N1[...,0] - 0.5, 
            np.mean(T_N1[...,1:n1], -1) - 0.5
        ), -1)
    
def delta_mu(T_N1i, X_N2i, G_N2i, T_N1j, X_N2j, G_N2j):
    Zi = summary_mu(T_N1i, X_N2i, G_N2i); Zj = summary_mu(T_N1j, X_N2j, G_N2j)
    
    return np.sum(np.abs(Zi - Zj), axis=-1)

def delta_pi(X_N1i, G_N1i, X_N1j, G_N1j):
    Zi = summary_pi(X_N1i, G_N1i); Zj = summary_pi(X_N1j, G_N1j)
    
    return np.sum(np.abs(Zi - Zj), axis=-1)

def delta(T_N1i, G_N2i, T_N1j, G_N2j):
    Zi = summary(T_N1i, G_N2i); Zj = summary(T_N1j, G_N2j)
    
    return np.sum(np.abs(Zi - Zj), axis=-1)