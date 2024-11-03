import numpy as np
import numpy.random as random
import pandas as pd

from .Data import Data

class KernelEstimate:
    def __init__(self, fit, i0s, T0s, G0, lamdas, Ds):
        self.fit = fit
        
        self.i0s = i0s
        self.T0s = T0s
        self.G0 = G0
        
        self.lamdas = np.array(lamdas)
        self.Ds = Ds
        self.ws = np.exp(
            - lamdas.reshape(lamdas.shape+(1,)*(Ds.ndim-1))
            * Ds.reshape((self.fit.data.n_node,)+(1,)*lamdas.ndim+Ds.shape[1:])
        )

    def est(self):
        return np.sum(
            self.fit.xis.reshape((self.fit.data.n_node,)+(1,)*(self.lamdas.ndim+self.Ds.ndim-1))
            * self.ws, 0
        ) / np.sum(self.ws, 0)
    
    def bb_bst(self, hops=1, n_bst=100, tqdm=None, level_tqdm=0):
        if self.G0.dist is None:
            self.G0.get_dist()
            
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
            
        cov_fits = [
            self.fit.model.cov_model.fit(
                Data(None, None, Xs_i, self.fit.data.G)
            )
            for Xs_i in self.fit.rX(n_bst, np.arange(self.fit.data.n_node), self.fit.data.G)
        ]
        
        ms_bst = np.zeros((self.fit.data.n_node, n_bst))
        vps_bst = np.zeros((self.fit.data.n_node, n_bst))
        for j in tqdm(np.arange(self.fit.data.n_node), smoothing=0, 
                      desc='bst_j', leave=None, position=level_tqdm):
            Xs_bst = np.array([
                cov_fit.sample(self.fit.n_X, self.fit.data.G.N2(j), self.fit.data.G)
                for cov_fit in cov_fits
            ])

            ms_bst[j] = np.mean(
                self.fit.mu(self.fit.data.Ts[self.fit.data.G.N1(j)], Xs_bst, 
                            self.fit.data.G.sub(self.fit.data.G.N2(j))),
                -1
            )

            vps_bst[j] = np.mean(
                self.fit.pi(self.fit.data.Ts[self.fit.data.G.N1(j)], Xs_bst, 
                            self.fit.data.G.sub(self.fit.data.G.N2(j))),
                -1
            )
            
        xis_bst = (
            (self.fit.data.Ys[:,None] - self.fit.mus[:,None])
            * vps_bst / self.fit.pis[:,None]
            + ms_bst
        )
        
        phis = (
            (xis_bst.reshape(xis_bst.shape+(1,)*(self.ws.ndim-1))
             - self.est()) * self.ws[:,None]
        ) / np.sum(self.ws, 0)

        hops = np.array(hops)
        Ks = self.G0.n_node / np.mean(np.sum(self.G0.dist <= hops[...,None,None], -1),-1)
        Ks_all = self.G0.n_node / np.mean(np.sum(
            self.G0.dist <= np.arange(np.max(hops).astype(int)+1)[1:,None,None], -1
        ), -1)

        phis_bst = np.zeros((n_bst,)+hops.shape+phis.shape[2:])
        for i_bst in tqdm(range(n_bst), smoothing=0, 
                      desc='bst', leave=None, position=level_tqdm):
            id_smp = np.random.choice(self.G0.n_node, np.max(Ks.astype(int)), replace=True)
            bs_bst = np.logical_and(
                self.G0.dist[id_smp] <= hops[...,None,None], 
                np.sum(
                    Ks_all[:,None].astype(int) > np.arange(np.max(Ks).astype(int)), 0
                )[:,None] >= hops[...,None,None]
            )
            phis_bst[i_bst] = np.sum(
                np.sum(bs_bst, -2).reshape(hops.shape+(self.G0.n_node,)+(1,)*(phis.ndim-2))
                * phis[:,i_bst], hops.ndim
            ) * (Ks / Ks.astype(int)).reshape(hops.shape+(1,)*(phis.ndim-2))

        return phis_bst

    