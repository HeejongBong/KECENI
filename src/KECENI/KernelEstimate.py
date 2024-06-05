import numpy as np
import numpy.random as random
import pandas as pd

def parzen_kernel(x, bw=None, G=None, const=2, eps=0.05):
    x = np.array(x)

    if bw is None:
        bw = np.array(
            const * np.log(G.n_node) 
            / np.log(np.maximum(np.mean(np.sum(G.Adj, 0)), 1+eps))
        )
    else:
        bw = np.array(bw)
    
    z = x/bw.reshape(bw.shape+(1,)*x.ndim)
    w = np.zeros(z.shape)
    
    ind1 = (z <= 0.5)
    ind2 = (z > 0.5) & (z <= 1)
    
    w[ind1] = 1 - 6 * z[ind1]**2 * (1-z[ind1])
    w[ind2] = 2 * (1-z[ind2])**3
    
    return w

class KernelEstimate:
    def __init__(self, fit, i0, T0, G0, lamdas, hs, Ds, xis, wms, offsets=None):
        self.fit = fit
        
        self.i0 = i0
        self.T0 = T0
        self.G0 = G0

        self.lamdas = np.array(lamdas)
        self.hs = np.array(hs)
        
        self.Ds = Ds
        self.xis = xis

        self.ws = np.exp(
            - self.lamdas.reshape(self.lamdas.shape+(1,)*self.hs.ndim)
            * self.Ds.reshape((self.fit.data.n_node,)+(1,)*self.lamdas.ndim+self.hs.shape)
        )

        self.wms = wms
        self.offsets = offsets

    def est(self, sum_offset=False):
        if self.offsets is None or not sum_offset:
            offsets = 0
        else:
            offsets = self.offsets 
            
        return np.sum(
            self.xis.reshape((self.fit.data.n_node,)+(1,)*self.lamdas.ndim+self.hs.shape)
            * self.ws + offsets, 0
        ) / np.sum(self.ws, 0)
        
    def phis_eif(self):
        if self.offsets is None:
            offsets = 0
        else:
            offsets = self.offsets
            
        phis = (
            (self.xis.reshape((self.fit.data.n_node,)+(1,)*self.lamdas.ndim+self.hs.shape)
             - self.est()) * self.ws + offsets
        ) / np.sum(self.ws, 0)

        return phis

    def phis_del(self):
        phis = (
            (self.xis.reshape((self.fit.data.n_node,)+(1,)*self.lamdas.ndim+self.hs.shape)
             - self.est()) * (2 * self.ws - self.wms)
        ) / np.sum(self.ws, 0)

        return phis
        
    def mse_eif_hac(self, hac_kernel=parzen_kernel, abs=False, **kwargs):
        phis = self.phis_eif()

        if abs:
            return np.sum(np.abs(phis) * np.tensordot(
                hac_kernel(self.fit.data.G.dist, G=self.fit.data.G, **kwargs), np.abs(phis), axes=(-1,0)
            ), -phis.ndim)
        else:
            return np.sum(phis * np.tensordot(
                hac_kernel(self.fit.data.G.dist, G=self.fit.data.G, **kwargs), phis, axes=(-1,0)
            ), -phis.ndim)

    def ste_eif_hac(self, hac_kernel=parzen_kernel, abs=False, **kwargs):
        return np.sqrt(self.mse_eif_hac(abs=abs, hac_kernel=hac_kernel, **kwargs))

    def mse_del_hac(self, hac_kernel=parzen_kernel, abs=False, **kwargs):
        phis = self.phis_del()

        if abs:
            return np.sum(np.abs(phis) * np.tensordot(
                hac_kernel(self.fit.data.G.dist, G=self.fit.data.G, **kwargs), np.abs(phis), axes=(-1,0)
            ), -phis.ndim)
        else:
            return np.sum(phis * np.tensordot(
                hac_kernel(self.fit.data.G.dist, G=self.fit.data.G, **kwargs), phis, axes=(-1,0)
            ), -phis.ndim)

    def ste_del_hac(self, hac_kernel=parzen_kernel, abs=False, **kwargs):
        return np.sqrt(self.mse_del_hac(abs=abs, hac_kernel=hac_kernel, **kwargs))

    def bb_bst_eif(self, hops=1, n_bst=1000, tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
            
        hops = np.array(hops)
        Ks = self.fit.data.n_node / np.mean(np.sum(self.fit.data.G.dist <= hops[...,None,None], -1),-1)
        Ks_all = self.fit.data.n_node / np.mean(np.sum(
            self.fit.data.G.dist <= np.arange(np.max(hops).astype(int)+1)[1:,None,None], -1
        ), -1)

        phis = self.phis_eif()

        phis_bst = np.zeros((n_bst,)+hops.shape+phis.shape[1:])
        for i_bst in tqdm(range(n_bst), smoothing=0, desc='bst', leave=None, position=level_tqdm):
            id_smp = np.random.choice(self.fit.data.n_node, np.max(Ks.astype(int)), replace=True)
            bs_bst = np.logical_and(
                self.fit.data.G.dist[id_smp] <= hops[...,None,None], 
                np.sum(
                    Ks_all[:,None].astype(int) > np.arange(np.max(Ks).astype(int)), 0
                )[:,None] >= hops[...,None,None]
            )
            phis_bst[i_bst] = np.sum(
                np.sum(bs_bst, -2).reshape(hops.shape+(self.fit.data.n_node,)+(1,)*(phis.ndim-1))
                * phis, hops.ndim
            ) * (Ks / Ks.astype(int)).reshape(hops.shape+(1,)*(phis.ndim-1))

        return phis_bst

    def mse_eif_bbb(self, hops=1, n_bst=1000, tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
                
        phis_bst = self.bb_bst_eif(hops, n_bst, tqdm, level_tqdm)
        
        return np.var(phis_bst, 0)

    def ste_eif_bbb(self, hops=1, n_bst=1000, tqdm=None, level_tqdm=0):
        return np.sqrt(self.mse_eif_bbb(hops, n_bst, tqdm, level_tqdm))

    def bb_bst_del(self, hops=1, n_bst=1000, tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
            
        hops = np.array(hops)
        Ks = self.fit.data.n_node / np.mean(np.sum(self.fit.data.G.dist <= hops[...,None,None], -1),-1)
        Ks_all = self.fit.data.n_node / np.mean(np.sum(
            self.fit.data.G.dist <= np.arange(np.max(hops).astype(int)+1)[1:,None,None], -1
        ), -1)

        phis = self.phis_del()

        phis_bst = np.zeros((n_bst,)+hops.shape+phis.shape[1:])
        for i_bst in tqdm(range(n_bst), smoothing=0, desc='bst', leave=None, position=level_tqdm):
            id_smp = np.random.choice(self.fit.data.n_node, np.max(Ks.astype(int)), replace=True)
            bs_bst = np.logical_and(
                self.fit.data.G.dist[id_smp] <= hops[...,None,None], 
                np.sum(
                    Ks_all[:,None].astype(int) > np.arange(np.max(Ks).astype(int)), 0
                )[:,None] >= hops[...,None,None]
            )
            phis_bst[i_bst] = np.sum(
                np.sum(bs_bst, -2).reshape(hops.shape+(self.fit.data.n_node,)+(1,)*(phis.ndim-1))
                * phis, hops.ndim
            ) * (Ks / Ks.astype(int)).reshape(hops.shape+(1,)*(phis.ndim-1))

        return phis_bst

    def mse_del_bbb(self, hops=1, n_bst=1000, tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
                
        phis_bst = self.bb_bst_del(hops, n_bst, tqdm, level_tqdm)
        
        return np.var(phis_bst, 0)

    def ste_del_bbb(self, hops=1, n_bst=1000, tqdm=None, level_tqdm=0):
        return np.sqrt(self.mse_del_bbb(hops, n_bst, tqdm, level_tqdm))