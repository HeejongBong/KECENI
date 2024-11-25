import numpy as np
import numpy.random as random
import pandas as pd

from . import istarmap
from .Data import Data
from .IT_broadcaster import IT_broadcaster

class KernelEstimate:
    def __init__(self, fit, i0s, T0s, G0, n_X, js, Ds, mus, pis, ms, vps):
        self.fit = fit
        
        self.i0s = i0s
        self.T0s = T0s
        self.G0 = G0
        
        self.n_X = n_X
        self.js = js
        
        self.Ds = Ds
        
        self.mus = mus
        self.pis = pis
        self.ms = ms
        self.vps = vps
        
        self.xis = (self.fit.data.Ys[js] - self.mus) * self.vps / self.pis + self.ms
        
    def ws(self, lamdas):
        lamdas = np.array(lamdas)
        
        return np.exp(
            - lamdas.reshape(lamdas.shape+(1,)*(self.Ds.ndim-1))
            * self.Ds.reshape((len(self.js),)+(1,)*lamdas.ndim+self.Ds.shape[1:])
        )
    
    def D_cv_j(self, j, i0s):
#         def AIPW_j(self, j, i0s, T0s=None, G0=None, n_X=100, seed=12345):
#         np.random.seed(seed)

        G0 = self.fit.data.G
        T0s = self.fit.data.Ts
        nin = np.logical_not(np.isin(i0s, self.fit.data.G.N2(j)))
        
        T_N1j = self.fit.data.Ts[self.fit.data.G.N1(j)]
            
        ITb = IT_broadcaster(i0s[nin], T0s)
        
        Ds_N2j = np.zeros(ITb.b.size)
        for ix, (i0, T0) in enumerate(ITb):
            T0_N1i0 = T0[G0.N1(i0)]
            Ds_N2j[ix] = self.fit.model.delta(
                T_N1j, self.fit.data.G.sub(self.fit.data.G.N2(j)),
                T0_N1i0, G0.sub(G0.N2(i0))
            )
                
        # def D_j(self, j, i0s, T0s=None, G0=None):

        D = np.full(i0s.shape, np.inf)
        D[...,nin] = Ds_N2j

        return D

    def cv(self, i0s=None, n_process=1, tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
            
        if i0s is None:
            i0s = np.arange(self.fit.data.n_node)
                
        # def D_cv_j(self, j, i0s):

        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.D_cv_j, map(
                lambda j: (j, i0s), self.js
            )), total=len(self.js), leave=None, position=level_tqdm, desc='j', smoothing=0))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.D_cv_j, map(
                    lambda j: (j, i0s), self.js
                )), total=len(self.js), leave=None, position=level_tqdm, desc='j', smoothing=0))

        Ds_cv = np.array(r)
        
        return CrossValidationEstimate(
            self.fit, i0s, self.fit.data.Ts, self.fit.data.G, self.n_X, self.js, 
            Ds_cv, self.mus, self.pis, self.ms, self.vps
        )
        
    def est(self, lamdas):
        lamdas = np.array(lamdas)
        ws = self.ws(lamdas)
        
        return np.sum(
            self.xis.reshape((len(self.js),)+(1,)*(lamdas.ndim+self.Ds.ndim-1))
            * ws, 0
        ) / np.sum(ws, 0)
    
    def nuisance_bst_j(self, j, Xs_bst):
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
        
    def phis_eif(self, lamdas, n_bst=100, cov_bst=None, n_process=1, tqdm=None, level_tqdm=0):
        lamdas = np.array(lamdas)
        ws = self.ws(lamdas)
            
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
    
        if cov_bst is None:
            cov_bst = self.fit.cov_bst(n_bst=n_bst)
        else:
            n_bst = len(cov_bst)
            
        # def nuisance_bst_j(self, j, Xs_bst):
        
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.nuisance_bst_j, map(
                lambda j: (j, np.array([
                    cov_fit.sample(self.n_X, self.fit.data.G.N2(j), self.fit.data.G)
                    for cov_fit in cov_bst
                ])), self.js
            )), total=len(self.js), leave=None, position=level_tqdm, desc='j', smoothing=0))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.nuisance_bst_j, map(
                    lambda j: (j, np.array([
                        cov_fit.sample(self.n_X, self.fit.data.G.N2(j), self.fit.data.G)
                        for cov_fit in cov_bst
                    ])), self.js
                )), total=len(self.js), leave=None, position=level_tqdm, desc='j', smoothing=0))
                
        ms_bst, vps_bst = list(zip(*r))
            
        xis_bst = (
            (self.fit.data.Ys[self.js,None] - self.mus[:,None])
            * vps_bst / self.pis[:,None]
            + ms_bst
        )
        
        phis = (
            (xis_bst.reshape(xis_bst.shape+(1,)*(ws.ndim-1))
             - self.est(lamdas)) * ws[:,None]
        ) / np.sum(ws, 0)

        return phis

    def nuisance_with_residual_j(self, k, j, Xs_bst):
        _, res_mu = self.fit.mu_fit.predict_with_residual(
            self.fit.data.Ts[self.fit.data.G.N1(j)], 
            self.fit.data.Xs[self.fit.data.G.N2(j)],
            self.fit.data.G.sub(self.fit.data.G.N2(j))
        )
        
        mus_bst, res_mus = self.fit.mu_fit.predict_with_residual(
            self.fit.data.Ts[self.fit.data.G.N1(j)], Xs_bst, 
            self.fit.data.G.sub(self.fit.data.G.N2(j))
        )
        
        ms_bst = np.mean(mus_bst, 1)
        res_ms = np.mean(res_mus, 1)
        
        _, res_pi = self.fit.pi_fit.predict_with_residual(
            self.fit.data.Ts[self.fit.data.G.N1(j)], 
            self.fit.data.Xs[self.fit.data.G.N2(j)],
            self.fit.data.G.sub(self.fit.data.G.N2(j))
        )
        
        pis_bst, res_pis = self.fit.pi_fit.predict_with_residual(
            self.fit.data.Ts[self.fit.data.G.N1(j)], Xs_bst, 
            self.fit.data.G.sub(self.fit.data.G.N2(j))
        )
        
        vps_bst = np.mean(pis_bst, 1)
        res_vps = np.mean(res_pis, 1)
        
        res_bst = (
            - vps_bst[:,None] / (self.pis[k]**2) * (self.fit.data.Ys[j] - self.mus[k]) 
            * res_pi
            + 1 / self.pis[k] * (self.fit.data.Ys[j] - self.mus[k]) 
            * res_vps
            - vps_bst[:,None] / self.pis[k] * res_mu
            + res_ms
        )
        
        return ms_bst, vps_bst, res_bst
        
    
    def phis_eif_with_offsets(self, lamdas, n_bst=100, cov_bst=None, tqdm=None, level_tqdm=0):
        lamdas = np.array(lamdas)
        ws = self.ws(lamdas)
            
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
            
        if cov_bst is None:
            cov_bst = self.fit.cov_bst(n_bst=n_bst)
        else:
            n_bst = len(cov_bst)

        # def nuisance_with_residual_j(self, j, Xs_bst):
        
        ms_bst = []; vps_bst = []
        offsets = np.zeros((n_bst,self.fit.data.n_node) + ws.shape[1:])
        for k, j in tqdm(enumerate(self.js), total=len(self.js), leave=None, 
                      position=level_tqdm, desc='j', smoothing=0):
            ms_j, vps_j, res_j = self.nuisance_with_residual_j(
                k, j, np.array([
                    cov_fit.sample(self.n_X, self.fit.data.G.N2(j), self.fit.data.G)
                    for cov_fit in cov_bst
                ])
            )
            
            ms_bst.append(ms_j); vps_bst.append(vps_j)
            offsets += ws[k] * res_j.reshape(res_j.shape + (1,) * (ws.ndim-1))
        
        ms_bst = np.array(ms_bst); vps_bst = np.array(vps_bst)
        offsets = offsets / np.sum(ws, 0)
        
        xis_bst = (
            (self.fit.data.Ys[self.js,None] - self.mus[:,None])
            * vps_bst / self.pis[:,None]
            + ms_bst
        )
        
        phis = (
            (xis_bst.reshape(xis_bst.shape+(1,)*(ws.ndim-1))
             - self.est(lamdas)) * ws[:,None]
        ) / np.sum(ws, 0)
        
        return phis, np.moveaxis(offsets, 1, 0)
    
    def phis_dr(self, lamdas, n_bst=100, cov_bst=None, tqdm=None, level_tqdm=0):
        phis, offsets = self.phis_eif_with_offsets(
            lamdas, n_bst, cov_bst, tqdm, level_tqdm
        )
        
        return phis + offsets
    
class CrossValidationEstimate(KernelEstimate):
    def xs_xhs(self, lamdas):
        _, id_xs, id_xhs = np.intersect1d(self.js, self.i0s, return_indices=True)
        return self.xis[id_xs], self.est(lamdas)[...,id_xhs]
    
def concat_KEs(list_KE):
    return KernelEstimate(
        list_KE[0].fit, list_KE[0].i0s, list_KE[0].T0s, list_KE[0].G0,
        np.min([KE_i.n_X for KE_i in list_KE]),
        np.concatenate([KE_i.js for KE_i in list_KE]),
        np.concatenate([KE_i.Ds for KE_i in list_KE]),
        np.concatenate([KE_i.mus for KE_i in list_KE]),
        np.concatenate([KE_i.pis for KE_i in list_KE]),
        np.concatenate([KE_i.ms for KE_i in list_KE]),
        np.concatenate([KE_i.vps for KE_i in list_KE]),
    )

def concat_CVs(list_CV):
    return CrossValidationEstimate(
        list_CV[0].fit, list_CV[0].i0s, list_CV[0].T0s, list_CV[0].G0,
        np.min([CV_i.n_X for CV_i in list_CV]),
        np.concatenate([CV_i.js for CV_i in list_CV]),
        np.concatenate([CV_i.Ds for CV_i in list_CV]),
        np.concatenate([CV_i.mus for CV_i in list_CV]),
        np.concatenate([CV_i.pis for CV_i in list_CV]),
        np.concatenate([CV_i.ms for CV_i in list_CV]),
        np.concatenate([CV_i.vps for CV_i in list_CV]),
    )

def concat_phis(list_ws, list_phis, list_offsets=None):
    if list_offsets is None:
        return np.concatenate([
            phis_i * ws_i / np.sum(list_ws, 0)
            for phis_i, ws_i in zip(list_phis, list_ws)
        ])
    else:
        phis_eif = concat_phis(list_ws, list_phis)
        return (
            phis_eif + np.sum(np.moveaxis(list_offsets, 0, -3) * np.array(list_ws), -3)
            / np.sum(list_ws, 0)
        )
