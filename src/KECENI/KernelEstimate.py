import numpy as np
import numpy.random as random
import pandas as pd

from . import istarmap
from .Data import Data
from .IT_broadcaster import IT_broadcaster

class KernelEstimate:
    def __init__(self, fit, i0s, T0s, G0, Ds):
        self.fit = fit
        
        self.i0s = i0s
        self.T0s = T0s
        self.G0 = G0
        
        self.Ds = Ds
        
    def ws(self, lamdas):
        lamdas = np.array(lamdas)
        
        return np.exp(
            - lamdas.reshape(lamdas.shape+(1,)*(self.Ds.ndim-1))
            * self.Ds.reshape((len(self.fit.js),)+(1,)*lamdas.ndim+self.Ds.shape[1:])
        )
        
    def est(self, lamdas):
        lamdas = np.array(lamdas)
        ws = self.ws(lamdas)
        
        return np.sum(
            self.fit.xis.reshape((len(self.fit.js),)+(1,)*(lamdas.ndim+self.Ds.ndim-1))
            * ws, 0
        ) / np.sum(ws, 0)
    
    def phis_eif(self, lamdas):
        lamdas = np.array(lamdas)
        ws = self.ws(lamdas)
        
        phis = (
            (self.fit.xis.reshape(self.fit.xis.shape+(1,)*(ws.ndim-1))
             - self.est(lamdas)) * ws
        ) / np.sum(ws, 0)
        
        return phis
        
    
    def nus_bst_j(self, j, Xs_bst):
        ms_bst = np.mean(
            self.fit.mu(self.fit.data.Ts[self.fit.data.G.N1(j)], Xs_bst, 
                        self.fit.data.G.sub(self.fit.data.G.N2(j))),
            -1
        )
        
        vps_bst = np.mean(
            self.fit.pi(self.fit.data.Ts[self.fit.data.G.N1(j)], Xs_bst, 
                        self.fit.data.G.sub(self.fit.data.G.N2(j))),
            -1
        )
        
        return ms_bst, vps_bst
        
    def phis_bst(self, lamdas, n_bst=100, cov_bst=None, n_process=1, tqdm=None, level_tqdm=0):
        lamdas = np.array(lamdas)
        ws = self.ws(lamdas)
            
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
    
        if cov_bst is None:
            cov_bst = self.fit.cov_bst(n_bst=n_bst)
        else:
            n_bst = len(cov_bst)
            
        # def nus_bst_j(self, j, Xs_bst):
        
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.nus_bst_j, map(
                lambda j: (j, np.array([
                    cov_fit.sample(self.fit.n_X, self.fit.data.G.N2(j), self.fit.data.G)
                    for cov_fit in cov_bst
                ])), self.fit.js
            )), total=len(self.fit.js), leave=None, position=level_tqdm, desc='j', smoothing=0))
        
        elif n_process is None or n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.nus_bst_j, map(
                    lambda j: (j, np.array([
                        cov_fit.sample(self.fit.n_X, self.fit.data.G.N2(j), self.fit.data.G)
                        for cov_fit in cov_bst
                    ])), self.fit.js
                )), total=len(self.fit.js), leave=None, position=level_tqdm, desc='j', smoothing=0))
                
        ms_bst, vps_bst = list(zip(*r))
            
        xis_bst = (
            (self.fit.data.Ys[self.fit.js,None] - self.fit.mus[:,None])
            * vps_bst / self.fit.pis[:,None]
            + ms_bst
        )
        
        phis = (
            (xis_bst.reshape(xis_bst.shape+(1,)*(ws.ndim-1))
             - self.est(lamdas)) * ws[:,None]
        ) / np.sum(ws, 0)

        return phis
    
    def H_nu_j(self, k, j, n_X):
        Xs_bst = self.fit.rX(n_X, self.fit.data.G.N2(j), self.fit.data.G)
        
        H_mu = self.fit.mu_fit.H(
            self.fit.data.Ts[self.fit.data.G.N1(j)], 
            self.fit.data.Xs[self.fit.data.G.N2(j)],
            self.fit.data.G.sub(self.fit.data.G.N2(j))
        )
        
        H_mus_bst = self.fit.mu_fit.H(
            self.fit.data.Ts[self.fit.data.G.N1(j)], Xs_bst, 
            self.fit.data.G.sub(self.fit.data.G.N2(j))
        )
        
        H_m = np.mean(H_mus_bst, -2)
        
        H_pi = self.fit.pi_fit.H(
            self.fit.data.Ts[self.fit.data.G.N1(j)], 
            self.fit.data.Xs[self.fit.data.G.N2(j)],
            self.fit.data.G.sub(self.fit.data.G.N2(j))
        )
        
        H_pis_bst = self.fit.pi_fit.H(
            self.fit.data.Ts[self.fit.data.G.N1(j)], Xs_bst, 
            self.fit.data.G.sub(self.fit.data.G.N2(j))
        )
        
        H_vp = np.mean(H_pis_bst, -2)
        
        H_nu = (
            - self.fit.vps[k] / (self.fit.pis[k]**2) * (self.fit.data.Ys[j] - self.fit.mus[k]) 
            * H_pi
            + 1 / self.fit.pis[k] * (self.fit.data.Ys[j] - self.fit.mus[k]) 
            * H_vp
            - self.fit.vps[k] / self.fit.pis[k] * H_mu
            + H_m
        )
        
        return H_nu

    def Hs_nu(self, lamdas, n_X=None, tqdm=None, level_tqdm=0):
        lamdas = np.array(lamdas)
        ws = self.ws(lamdas)
            
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
            
        if n_X is None:
            n_X = self.fit.n_X

        # def H_nu_j(self, k, j, Xs_bst):
        wHs_nu = np.zeros((self.fit.data.n_node,) + ws.shape[1:])
        for k, j in tqdm(enumerate(self.fit.js), total=len(self.fit.js), leave=None, 
                         position=level_tqdm, desc='j', smoothing=0):
            wHs_nu += ws[k] * self.H_nu_j(
                k, j, n_X
            ).reshape((self.fit.data.n_node,) + (1,) * (ws.ndim-1))
        
        Hs_nu = wHs_nu / np.sum(ws, 0)
        
        return Hs_nu
    
    def phis_bst_dr(self, lamdas, n_bst=100, cov_bst=None, n_X=None, 
                    n_process=1, tqdm=None, level_tqdm=0):
        # def phis_bst(self, lamdas, n_bst=100, cov_bst=None, n_process=1, tqdm=None, level_tqdm=0):
        phis_bst = self.phis_bst(lamdas, n_bst, cov_bst, n_process, tqdm, level_tqdm)
        
        # def Hs_nu(self, lamdas, n_X=None, tqdm=None, level_tqdm=0):
        Hs_nu = self.Hs_nu(lamdas, n_X, tqdm, level_tqdm)
        
        return phis_bst + Hs_nu[:,None]
    
    def H_px_j(self, k, j, n_sample):
        H_px_m = self.fit.cov_fit.H(lambda X_N2: self.fit.mu(
            self.fit.data.Ts[self.fit.data.G.N1(j)],
            X_N2, self.fit.data.G.sub(self.fit.data.G.N2(j))
        ), n_sample, self.fit.data.G.N2(j), self.fit.data.G)

        H_px_vp = self.fit.cov_fit.H(lambda X_N2: self.fit.pi(
            self.fit.data.Ts[self.fit.data.G.N1(j)],
            X_N2, self.fit.data.G.sub(self.fit.data.G.N2(j))
        ), n_sample, self.fit.data.G.N2(j), self.fit.data.G)
        
        H_px = (
            1 / self.fit.pis[k] * (self.fit.data.Ys[j] - self.fit.mus[k]) 
            * H_px_vp
            + H_px_m
        )
        
        return H_px
    
    def H_j(self, k, j, n_X, n_sample):
        H_nu = self.H_nu_j(k, j, n_X)
        
        H_px = self.H_px_j(k, j, n_sample)
        
        return H_nu + H_px
    
    def Hs_px(self, lamdas, n_sample=None, tqdm=None, level_tqdm=0):
        lamdas = np.array(lamdas)
        ws = self.ws(lamdas)
            
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
            
        if n_sample is None:
            n_sample = self.fit.n_X

        # def H_j(self, k, j, Xs_bst, n_sample):
        wHs = np.zeros((self.fit.data.n_node,) + ws.shape[1:])
        for k, j in tqdm(enumerate(self.fit.js), total=len(self.fit.js), leave=None, 
                      position=level_tqdm, desc='j', smoothing=0):
            wHs += ws[k] * self.H_px_j(
                k, j, n_sample
            ).reshape((self.fit.data.n_node,) + (1,) * (ws.ndim-1))
        
        Hs = wHs / np.sum(ws, 0)
        return Hs
    
    def Hs(self, lamdas, n_X=None, n_sample=None, tqdm=None, level_tqdm=0):
        lamdas = np.array(lamdas)
        ws = self.ws(lamdas)
            
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
            
        if n_X is None:
            n_X = self.fit.n_X
            
        if n_sample is None:
            n_sample = n_X

        # def H_j(self, k, j, Xs_bst, n_sample):
        wHs = np.zeros((self.fit.data.n_node,) + ws.shape[1:])
        for k, j in tqdm(enumerate(self.fit.js), total=len(self.fit.js), leave=None, 
                      position=level_tqdm, desc='j', smoothing=0):
            wHs += ws[k] * self.H_j(
                k, j, n_X, n_sample
            ).reshape((self.fit.data.n_node,) + (1,) * (ws.ndim-1))
        
        Hs = wHs / np.sum(ws, 0)
        return Hs
    
    def phis_sdw_dr(self, lamdas, n_X=None, n_sample=None, tqdm=None, level_tqdm=0):
        # def phis_eif(self, lamdas):
        phis_eif = self.phis_eif(lamdas)
        
        # def Hs(self, lamdas, n_X=None, n_sample=None, tqdm=None, level_tqdm=0):
        Hs = self.Hs(lamdas, n_X, n_sample, tqdm, level_tqdm)
        
        return phis_eif + Hs
    
    def sub(self, ind_js):
        # def __init__(self, fit, i0s, T0s, G0, Ds):
        
        return KernelEstimate(
            self.fit.sub(ind_js), self.i0s, self.T0s, self.G0, self.Ds[ind_js]
        )
        
    
class CrossValidationEstimate(KernelEstimate):
    def xs_xhs(self, lamdas):
        _, id_xs, id_xhs = np.intersect1d(self.fit.js, self.i0s, return_indices=True)
        return self.fit.xis[id_xs], self.est(lamdas)[...,id_xhs]
    
    def sub(self, ind_js):
        # def __init__(self, fit, i0s, T0s, G0, Ds):
        
        return CrossValidationEstimate(
            self.fit.sub(ind_js), self.i0s, self.T0s, self.G0, self.Ds[ind_js]
        )
   
