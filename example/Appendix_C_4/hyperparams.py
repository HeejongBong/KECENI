import numpy as np
import scipy.linalg as la
import ot

from itertools import starmap

def avg_mu(T_N1, X_N2, G_N2):
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
    
def avg_pi(X_N1, G_N1):
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
    
def avg(T_N1, G_N2):
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

def wavg_mu(T_N1, X_N2, G_N2):
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
            np.mean((T_N1[...,1:n1] - 0.5) * np.sum(G_N2.Adj[1:n1,0:n1], -1), -1), 
            *np.moveaxis(X_N2[...,0,:], -1, 0), 
            *np.moveaxis(np.mean(X_N2[...,1:n1,:] * np.sum(G_N2.Adj[1:n1,0:n1], -1)[...,None], -2), -1, 0)
        ), -1)
    
def wavg_pi(X_N1, G_N1):
    n1 = X_N1.shape[-2]; d = X_N1.shape[-1]
        
    if n1 == 1:
        return np.stack(np.broadcast_arrays(
            *np.moveaxis(X_N1[...,0,:], -1, 0), 
            *np.moveaxis(np.full(X_N1[...,0,:].shape, 0), -1, 0)
        ), -1)
    else:
        return np.stack(np.broadcast_arrays(
            *np.moveaxis(X_N1[...,0,:], -1, 0), 
            *np.moveaxis(np.mean(X_N1[...,1:n1,:] * np.sum(G_N1.Adj[1:n1,0:n1], -1)[...,None], -2), -1, 0)
        ), -1)
    
def wavg(T_N1, G_N2):
    n1 = T_N1.shape[-1]
    
    if n1 == 1:
        return np.stack(np.broadcast_arrays(
            T_N1[...,0] - 0.5, 
            np.full(T_N1[...,0].shape, 0)
        ), -1)
    else:
        return np.stack(np.broadcast_arrays(
            T_N1[...,0] - 0.5, 
            np.mean((T_N1[...,1:n1] - 0.5) * np.sum(G_N2.Adj[1:n1,0:n1], -1), -1)
        ), -1)

class GW:
    def __init__(self, alpha_T=1, alpha_X=1):
        self.alpha_T = alpha_T
        self.alpha_X = alpha_X
                
    def delta(self, Ts_N1i, G_N2i, Ts_N1j, G_N2j):
        G_N1i = G_N2i.sub(G_N2i.N1(0)); G_N1j = G_N2j.sub(G_N2j.N1(0))        
        
        if G_N1i.n_node == 1:
            C_i = np.zeros(G_N1i.Adj.shape)
            To_i = np.zeros(Ts_N1i.shape)
        else:
            C_i = 1 - G_N1i.Adj[1:,1:]/2
            C_i[np.arange(G_N1i.n_node-1), np.arange(G_N1i.n_node-1)] = 0
            To_i = Ts_N1i[...,1:] - 0.5
            
        if G_N1j.n_node == 1:
            C_j = np.zeros(G_N1j.Adj.shape)
            To_j = np.zeros(Ts_N1j.shape)
        else:
            C_j = 1 - G_N1j.Adj[1:,1:]/2
            C_j[np.arange(G_N1j.n_node-1), np.arange(G_N1j.n_node-1)] = 0
            To_j = Ts_N1j[...,1:] - 0.5
                
        Ms_ij = self.alpha_T * (
            np.abs(
                To_i[...,:,None] - To_j[...,None,:]
            )
        )
                
        ds_ij = np.array(list(starmap(ot.fused_gromov_wasserstein2, map(
            lambda M_ij: (M_ij, C_i, C_j),
            Ms_ij.reshape((-1,)+Ms_ij.shape[-2:])
        )))).reshape(Ms_ij.shape[:-2])

        return (
            self.alpha_T * np.abs(Ts_N1i[...,0] - Ts_N1j[...,0])
            + ds_ij
        )

    def delta_mu(self, Ts_N1i, Xs_N2i, G_N2i, Ts_N1j, Xs_N2j, G_N2j):
        G_N1i = G_N2i.sub(G_N2i.N1(0)); G_N1j = G_N2j.sub(G_N2j.N1(0))
        Xs_N1i = Xs_N2i[...,G_N2i.N1(0),:]; Xs_N1j = Xs_N2j[...,G_N2j.N1(0),:]
        
        if G_N1i.n_node == 1:
            C_i = np.zeros(G_N1i.Adj.shape)
            To_i = np.zeros(Ts_N1i.shape)
            Xo_i = np.zeros(Xs_N1i.shape)
        else:
            C_i = 1 - G_N1i.Adj[1:,1:]/2
            C_i[np.arange(G_N1i.n_node-1), np.arange(G_N1i.n_node-1)] = 0
            To_i = Ts_N1i[...,1:] - 0.5
            Xo_i = Xs_N1i[...,1:,:] - 0.5
            
        if G_N1j.n_node == 1:
            C_j = np.zeros(G_N1j.Adj.shape)
            To_j = np.zeros(Ts_N1j.shape)
            Xo_j = np.zeros(Xs_N1j.shape)
        else:
            C_j = 1 - G_N1j.Adj[1:,1:]/2
            C_j[np.arange(G_N1j.n_node-1), np.arange(G_N1j.n_node-1)] = 0
            To_j = Ts_N1j[...,1:] - 0.5
            Xo_j = Xs_N1j[...,1:,:] - 0.5
        
        Ms_ij = (
            self.alpha_T * np.abs(
                To_i[...,:,None] - To_j[...,None,:]
            )
            + self.alpha_X * np.sum(np.abs(
                Xo_i[...,:,None,:] - Xo_j[...,None,:,:]
            ), -1)
        )
                
        ds_ij = np.array(list(starmap(ot.fused_gromov_wasserstein2, map(
            lambda M_ij: (M_ij, C_i, C_j),
            Ms_ij.reshape((-1,)+Ms_ij.shape[-2:])
        )))).reshape(Ms_ij.shape[:-2])


        return (
            self.alpha_T * np.abs(Ts_N1i[...,0] - Ts_N1j[...,0])
            + self.alpha_X * np.sum(np.abs(Xs_N1i[...,0,:] - Xs_N1j[...,0,:]), -1)
            + ds_ij
        )

    def delta_pi(self, Xs_N1i, G_N1i, Xs_N1j, G_N1j):
        if G_N1i.n_node == 1:
            C_i = np.zeros(G_N1i.Adj.shape)
            Xo_i = np.zeros(Xs_N1i.shape)
        else:
            C_i = 1 - G_N1i.Adj[1:,1:]/2
            C_i[np.arange(G_N1i.n_node-1), np.arange(G_N1i.n_node-1)] = 0
            Xo_i = Xs_N1i[...,1:,:] - 0.5
            
        if G_N1j.n_node == 1:
            C_j = np.zeros(G_N1j.Adj.shape)
            Xo_j = np.zeros(Xs_N1j.shape)
        else:
            C_j = 1 - G_N1j.Adj[1:,1:]/2
            C_j[np.arange(G_N1j.n_node-1), np.arange(G_N1j.n_node-1)] = 0
            Xo_j = Xs_N1j[...,1:,:] - 0.5
        
        Ms_ij = (
            self.alpha_X * np.sum(np.abs(
                Xo_i[...,:,None,:] - Xo_j[...,None,:,:]
            ), -1)
        )
        
        ds_ij = np.array(list(starmap(ot.fused_gromov_wasserstein2, map(
            lambda M_ij: (M_ij, C_i, C_j),
            Ms_ij.reshape((-1,)+Ms_ij.shape[-2:])
        )))).reshape(Ms_ij.shape[:-2])
        
        return (
            self.alpha_X * np.sum(np.abs(Xs_N1i[...,0,:] - Xs_N1j[...,0,:]), -1)
            + ds_ij
        )
    
class L1:
    def __init__(self, summary_mu, summary_pi, summary):
        self.summary_mu = summary_mu
        self.summary_pi = summary_pi
        self.summary = summary

    def delta_mu(self, T_N1i, X_N2i, G_N2i, T_N1j, X_N2j, G_N2j):
        Zi = self.summary_mu(T_N1i, X_N2i, G_N2i); Zj = self.summary_mu(T_N1j, X_N2j, G_N2j)
        
        return np.sum(np.abs(Zi - Zj), axis=-1)

    def delta_pi(self, X_N1i, G_N1i, X_N1j, G_N1j):
        Zi = self.summary_pi(X_N1i, G_N1i); Zj = self.summary_pi(X_N1j, G_N1j)
        
        return np.sum(np.abs(Zi - Zj), axis=-1)

    def delta(self, T_N1i, G_N2i, T_N1j, G_N2j):
        Zi = self.summary(T_N1i, G_N2i); Zj = self.summary(T_N1j, G_N2j)
        
        return np.sum(np.abs(Zi - Zj), axis=-1)