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
        
    def phis_bst(self, lamdas, xis_bst=None, n_bst=100, cov_bst=None, n_process=1, tqdm=None, level_tqdm=0):
        lamdas = np.array(lamdas)
        ws = self.ws(lamdas)
        est = self.est(lamdas)
        
        if xis_bst is None:
            xis_bst = self.fit.xis_bst(n_bst, cov_bst, n_process, tqdm, level_tqdm)
        
        phis = (
            (xis_bst.reshape(xis_bst.shape+(1,)*est.ndim) - est) 
            * ws[:,None]
        ) / np.sum(ws, 0)

        return phis
    
    def wH_nu(self, lamdas, Hs_nu=None, n_X=None, n_process=1, tqdm=None, level_tqdm=0):
        lamdas = np.array(lamdas)
        ws = self.ws(lamdas)
        
        if Hs_nu is None:
            if tqdm is None:
                def tqdm(iterable, *args, **kwargs):
                    return iterable

            if n_X is None:
                n_X = self.fit.n_X

            # def H_nu_j(self, k, j, Xs_bst):
            wH = sum((
                ws[k] * self.fit.H_nu_j(
                    k, j, n_X
                ).reshape((self.fit.data.n_node,) + (1,) * (ws.ndim-1))
                for k, j in tqdm(enumerate(self.fit.js), total=len(self.fit.js), leave=None,
                                 position = level_tqdm, desc='Hs_nu', smoothing=0)
            ))
            
        else:
            wH = np.sum(Hs_nu.reshape(Hs_nu.shape+(1,)*(ws.ndim-1)) * ws[:,None], 0)
        
        return wH / np.sum(ws, 0)

    def wH_px(self, lamdas, Hs_px=None, n_X=None, tqdm=None, level_tqdm=0):
        lamdas = np.array(lamdas)
        ws = self.ws(lamdas)
        
        if Hs_px is None:
            if tqdm is None:
                def tqdm(iterable, *args, **kwargs):
                    return iterable

            if n_X is None:
                n_X = self.fit.n_X

            # def H_px_j(self, k, j, n_X):
            wH = sum((
                ws[k] * self.fit.H_px_j(
                    k, j, n_X
                ).reshape((self.fit.data.n_node,) + (1,) * (ws.ndim-1))
                for k, j in tqdm(enumerate(self.fit.js), total=len(self.fit.js), leave=None,
                                 position = level_tqdm, desc='Hs_px', smoothing=0)
            ))
            
        else:
            wH = np.sum(Hs_px.reshape(Hs_px.shape+(1,)*(ws.ndim-1)) * ws[:,None], 0)

        return wH / np.sum(ws, 0)

#     def H_j(self, k, j, n_X, n_bst):
#         H_nu = self.H_nu_j(k, j, n_X)
        
#         H_px = self.H_px_j(k, j, n_X, n_bst)
        
#         return H_nu + H_px
    
#     def Hs(self, lamdas, n_X=None, n_bst=None, tqdm=None, level_tqdm=0):
#         lamdas = np.array(lamdas)
#         ws = self.ws(lamdas)
            
#         if tqdm is None:
#             def tqdm(iterable, *args, **kwargs):
#                 return iterable
            
#         if n_X is None:
#             n_X = self.fit.n_X
            
#         if n_bst is None:
#             n_bst = self.fit.data.n_node

#         # def H_j(self, k, j, n_X, n_bst):
#         wHs = np.zeros((self.fit.data.n_node,) + ws.shape[1:])
#         for k, j in tqdm(enumerate(self.fit.js), total=len(self.fit.js), leave=None, 
#                       position=level_tqdm, desc='j', smoothing=0):
#             wHs += ws[k] * self.H_j(
#                 k, j, n_X, n_bst
#             ).reshape((self.fit.data.n_node,) + (1,) * (ws.ndim-1))
        
#         Hs = wHs / np.sum(ws, 0)
#         return Hs
    
#     def phis_bst_dr(self, lamdas, n_bst=100, cov_bst=None, n_X=None, 
#                     n_process=1, tqdm=None, level_tqdm=0):
#         # def phis_bst(self, lamdas, n_bst=100, cov_bst=None, n_process=1, tqdm=None, level_tqdm=0):
#         phis_bst = self.phis_bst(lamdas, n_bst, cov_bst, n_process, tqdm, level_tqdm)
        
#         # def Hs_nu(self, lamdas, n_X=None, tqdm=None, level_tqdm=0):
#         Hs_nu = self.Hs_nu(lamdas, n_X, tqdm, level_tqdm)
        
#         return phis_bst + Hs_nu[:,None]

#     def phis_sdw_dr(self, lamdas, n_X=None, n_bst=None, tqdm=None, level_tqdm=0):
#         # def phis_eif(self, lamdas):
#         phis_eif = self.phis_eif(lamdas)
        
#         # def Hs(self, lamdas, n_X=None, n_bst=None, tqdm=None, level_tqdm=0):
#         Hs = self.Hs(lamdas, n_X, n_bst, tqdm, level_tqdm)
        
#         return phis_eif + Hs
    
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
   
