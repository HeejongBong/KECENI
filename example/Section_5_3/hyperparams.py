import numpy as np
import scipy.linalg as la
import ot

from itertools import starmap

def dist(dZ):
    return np.abs(dZ)

class GW:
    def __init__(self, alpha_T=1, alpha_X=1):
        self.alpha_T = alpha_T
        self.alpha_X = alpha_X
                
    def delta(self, Ts_N1i, G_N2i, Ts_N1j, G_N2j):
        G_N1i = G_N2i.sub(G_N2i.N1(0)); G_N1j = G_N2j.sub(G_N2j.N1(0))        
        
        if G_N1i.n_node == 1:
            To_i = np.zeros(Ts_N1i.shape)
        else:
            To_i = Ts_N1i[...,1:] - 0.5
            
        if G_N1j.n_node == 1:
            To_j = np.zeros(Ts_N1j.shape)
        else:
            To_j = Ts_N1j[...,1:] - 0.5
                
        Ms_ij = (
            np.abs(
                To_i[...,:,None] - To_j[...,None,:]
            )
        )
                
        ds_ij = np.array(list(starmap(ot.emd2, map(
            lambda M_ij: (list(), list(), M_ij),
            Ms_ij.reshape((-1,)+Ms_ij.shape[-2:])
        )))).reshape(Ms_ij.shape[:-2])

        return (
            np.abs(Ts_N1i[...,0] - Ts_N1j[...,0])
            + ds_ij
        )

    def delta_mu(self, Ts_N1i, Xs_N2i, G_N2i, Ts_N1j, Xs_N2j, G_N2j):
        G_N1i = G_N2i.sub(G_N2i.N1(0)); G_N1j = G_N2j.sub(G_N2j.N1(0))
        Xs_N1i = Xs_N2i[...,G_N2i.N1(0),:]; Xs_N1j = Xs_N2j[...,G_N2j.N1(0),:]
        
        if G_N1i.n_node == 1:
            To_i = np.zeros(Ts_N1i.shape)
            Xo_i = np.zeros(Xs_N1i.shape)
        else:
            To_i = Ts_N1i[...,1:] - 0.5
            Xo_i = Xs_N1i[...,1:,:] - 0.5
            
        if G_N1j.n_node == 1:
            To_j = np.zeros(Ts_N1j.shape)
            Xo_j = np.zeros(Xs_N1j.shape)
        else:
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
                
        ds_ij = np.array(list(starmap(ot.emd2, map(
            lambda M_ij: (list(), list(), M_ij),
            Ms_ij.reshape((-1,)+Ms_ij.shape[-2:])
        )))).reshape(Ms_ij.shape[:-2])


        return (
            self.alpha_T * np.abs(Ts_N1i[...,0] - Ts_N1j[...,0])
            + self.alpha_X * np.sum(np.abs(Xs_N1i[...,0,:] - Xs_N1j[...,0,:]), -1)
            + ds_ij
        )

    def delta_pi(self, Xs_N1i, G_N1i, Xs_N1j, G_N1j):
        if G_N1i.n_node == 1:
            Xo_i = np.zeros(Xs_N1i.shape)
        else:
            Xo_i = Xs_N1i[...,1:,:] - 0.5
            
        if G_N1j.n_node == 1:
            Xo_j = np.zeros(Xs_N1j.shape)
        else:
            Xo_j = Xs_N1j[...,1:,:] - 0.5
        
        Ms_ij = (
            np.sum(np.abs(
                Xo_i[...,:,None,:] - Xo_j[...,None,:,:]
            ), -1)
        )
        
        ds_ij = np.array(list(starmap(ot.emd2, map(
            lambda M_ij: (list(), list(), M_ij),
            Ms_ij.reshape((-1,)+Ms_ij.shape[-2:])
        )))).reshape(Ms_ij.shape[:-2])
        
        return (
            np.sum(np.abs(Xs_N1i[...,0,:] - Xs_N1j[...,0,:]), -1)
            + ds_ij
        )