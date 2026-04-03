import numpy as np
import numpy.random as random
import pandas as pd

from . import istarmap
from .Data import Data
from .IT_broadcaster import IT_broadcaster

def box_kernel(x):
    return (np.abs(x) <= 1).astype(float)

def triangular_kernel(x):
    return np.maximum(1 - np.abs(x), 0)

def gaussian_kernel(x):
    return np.exp(-0.5 * x**2)

def epanechnikov_kernel(x):
    return np.maximum(1 - x**2, 0)

def exponential_kernel(x):
    return np.exp(-np.abs(x))

class KernelEstimate:
    def __init__(self, fit, i0s, T0s, G0, Ds):
        self.fit = fit
        
        self.i0s = i0s
        self.T0s = T0s
        self.G0 = G0
        
        self.Ds = Ds
        
    def ws(self, lamdas, kernel=None):
        lamdas = np.array(lamdas)

        if kernel is None:
            kernel = exponential_kernel
        
        return kernel(
            lamdas.reshape(lamdas.shape+(1,)*(self.Ds.ndim-1))
            * self.Ds.reshape((len(self.fit.js),)+(1,)*lamdas.ndim+self.Ds.shape[1:])
        )
        
    def est(self, lamdas, kernel=None):
        lamdas = np.array(lamdas)
        ws = self.ws(lamdas, kernel)
        
        return np.sum(
            self.fit.xis.reshape((len(self.fit.js),)+(1,)*(lamdas.ndim+self.Ds.ndim-1))
            * ws, 0
        ) / np.sum(ws, 0)
    
    def phis_eif(self, lamdas, kernel=None):
        lamdas = np.array(lamdas)
        ws = self.ws(lamdas, kernel)
        
        phis = (
            (self.fit.xis.reshape(self.fit.xis.shape+(1,)*(ws.ndim-1))
             - self.est(lamdas, kernel)) * ws
        ) / np.sum(ws, 0)
        
        return phis
        
    def phis_bst(self, lamdas, kernel=None, xis_bst=None, n_bst=100, cov_bst=None, n_process=1, tqdm=None, level_tqdm=0):
        lamdas = np.array(lamdas)
        ws = self.ws(lamdas, kernel)
        est = self.est(lamdas, kernel)
        
        if xis_bst is None:
            xis_bst = self.fit.xis_bst(n_bst, cov_bst, n_process, tqdm, level_tqdm)
        
        phis = (
            (xis_bst.reshape(xis_bst.shape+(1,)*est.ndim) - est) 
            * ws[:,None]
        ) / np.sum(ws, 0)

        return phis
    
    def wH_nu(self, lamdas, kernel=None, Hs_nu=None, n_X=None, tqdm=None, level_tqdm=0):
        lamdas = np.array(lamdas)
        ws = self.ws(lamdas, kernel)
        
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

    def wH_px(self, lamdas, kernel=None, Hs_px=None, n_X=None, tqdm=None, level_tqdm=0):
        lamdas = np.array(lamdas)
        ws = self.ws(lamdas, kernel)
        
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
    
    def sub(self, ind_js):
        # def __init__(self, fit, i0s, T0s, G0, Ds):
        
        return KernelEstimate(
            self.fit.sub(ind_js), self.i0s, self.T0s, self.G0, self.Ds[ind_js]
        )
        
    
class CrossValidationEstimate(KernelEstimate):
    def xs_xhs(self, lamdas, kernel=None):
        _, id_xs, id_xhs = np.intersect1d(self.fit.js, self.i0s, return_indices=True)
        return self.fit.xis[id_xs], self.est(lamdas, kernel)[...,id_xhs]
    
    def sub(self, ind_js):
        # def __init__(self, fit, i0s, T0s, G0, Ds):
        
        return CrossValidationEstimate(
            self.fit.sub(ind_js), self.i0s, self.T0s, self.G0, self.Ds[ind_js]
        )
   
