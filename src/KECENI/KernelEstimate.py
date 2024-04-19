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
    def __init__(self, fit, i0, T0, G0, lamdas, hs, Ds, xis, offsets=None):
        self.fit = fit
        
        self.i0 = i0
        self.T0 = T0
        self.G0 = G0

        self.lamdas = np.array(lamdas)
        self.hs = np.array(hs)
        
        self.Ds = Ds
        self.xis = xis

        self.offsets = offsets

    def est(self, sum_offset=False):
        if self.offsets is None or not sum_offset:
            offsets = 0
        else:
            offsets = self.offsets 
            
        return np.sum(
            self.xis.reshape((self.fit.data.n_node,)+(1,)*self.lamdas.ndim+self.hs.shape)
            * np.exp(- self.lamdas.reshape(self.lamdas.shape+(1,)*self.hs.ndim) 
                     * self.Ds.reshape((self.fit.data.n_node,)+(1,)*self.lamdas.ndim+self.hs.shape))
            + offsets, 0
        ) / np.sum(
            np.exp(- self.lamdas.reshape(self.lamdas.shape+(1,)*self.hs.ndim) 
                   * self.Ds.reshape((self.fit.data.n_node,)+(1,)*self.lamdas.ndim+self.hs.shape)), 0
        )

    def mse_hac(self, hac_kernel=parzen_kernel, abs=False, **kwargs):
        phis = self.get_phi()

        if abs:
            return np.sum(np.abs(phis) * np.tensordot(
                hac_kernel(self.fit.data.G.dist, G=self.fit.data.G, **kwargs), np.abs(phis), axes=(-1,0)
            ), -phis.ndim)
        else:
            return np.sum(phis * np.tensordot(
                hac_kernel(self.fit.data.G.dist, G=self.fit.data.G, **kwargs), phis, axes=(-1,0)
            ), -phis.ndim)

    def ste_hac(self, hac_kernel=parzen_kernel, abs=False, **kwargs):
        return np.sqrt(self.mse_hac(abs=abs, hac_kernel=hac_kernel, **kwargs))

    def bb_bst(self, hops=1, n_bst=1000, tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
            
        hops = np.array(hops)
        Ks = self.fit.data.n_node / np.mean(np.sum(self.fit.data.G.dist <= hops[...,None,None], -1),-1)
        Ks_all = self.fit.data.n_node / np.mean(np.sum(
            self.fit.data.G.dist <= np.arange(np.max(hops).astype(int)+1)[1:,None,None], -1
        ), -1)

        phis = self.get_phi()

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

    def mse_bbb(self, hops=1, n_bst=1000, tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
                
        phis_bst = self.bb_bst(hops, n_bst, tqdm, level_tqdm)
        
        return np.var(phis_bst, 0)

    def ste_bbb(self, hops=1, n_bst=1000, tqdm=None, level_tqdm=0):
        return np.sqrt(self.mse_bbb(hops, n_bst, tqdm, level_tqdm))

    def get_offset(self, lamdas=None, n_T=100, n_X=110, n_X0=None, n_process=1, tqdm=None, level_tqdm=0):
        if lamdas is None:
            lamdas = self.lamdas
        else:
            self.lamdas = np.array(lamdas)

        if n_T < 1:
            return None

        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.fit.EIF_j,
                (
                    (j, self.i0, self.T0, self.G0, lamdas, self.hs, 
                     n_T, n_X, n_X0, np.random.randint(12345))
                    for j in range(self.fit.data.n_node)
                )
            ), total=self.fit.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.fit.EIF_j,
                    (
                        (j, self.i0, self.T0, self.G0, lamdas, self.hs, 
                         n_T, n_X, n_X0, np.random.randint(12345))
                        for j in range(self.fit.data.n_node)
                    )
                ), total=self.fit.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0))

        Ds, xis, offsets = list(zip(*r))

        return offsets

    # def get_offset_archive(self, lamdas=None, n_sample=100, tqdm=None, level_tqdm=0):
    #     if tqdm is None:
    #         def tqdm(iterable, *args, **kwargs):
    #             return iterable

    #     if lamdas is None:
    #         lamdas = self.lamdas
    #     else:
    #         self.lamdas = lamdas
                
    #     hf = self.hs.flatten()

    #     T0_N1i0 = self.T0[self.G0.N1(self.i0)]
    #     G0_N2i0 = self.G0.sub(self.G0.N2(self.i0))

    #     offsets = list()
    #     for j in tqdm(range(self.fit.data.n_node), smoothing=0, desc='j', leave=None, position=level_tqdm):
    #         Xs_N2j = np.concatenate([
    #             self.fit.data.Xs[None,self.fit.data.G.N2(j)],
    #             self.fit.rX(n_sample-1, self.fit.data.G.N2(j), self.fit.data.G)
    #         ], 0)
    #         Ts_N1j = self.fit.rT(1, Xs_N2j, self.fit.data.G.sub(self.fit.data.G.N2(j)))[0]
            
    #         Ds_bst = list()
    #         ms_bst = list()
    #         mus_bst = list()
    #         nus_bst = list()
            
    #         for T_N1j in Ts_N1j:
    #             Xs_N2j = np.concatenate([
    #                 self.fit.data.Xs[None,self.fit.data.G.N2(j)],
    #                 self.fit.rX(n_sample-1, self.fit.data.G.N2(j), self.fit.data.G)
    #             ], 0)
    #             Xs_N2i0 = self.fit.rX(n_sample, self.G0.N2(self.i0), self.G0)
        
    #             Ds_N2j = self.fit.model.delta(
    #                 T0_N1i0[None,None], Xs_N2i0[:,None], 
    #                 self.G0.sub(self.G0.N2(self.i0)),
    #                 T_N1j[None,None], Xs_N2j[None,:], 
    #                 self.fit.data.G.sub(self.fit.data.G.N2(j))
    #             )
    #             mus_N2j = self.fit.mu(np.repeat(T_N1j[None,:], n_sample, 0), Xs_N2j, 
    #                                       self.fit.data.G.sub(self.fit.data.G.N2(j)))
                
    #             if self.fit.nu_method == 'ksm':
    #                 Ws_N2j = np.exp(- hf[...,None,None] 
    #                                 * (Ds_N2j - np.min(Ds_N2j, -1)[...,None]))
    #                 pnus_N2j = Ws_N2j / np.mean(Ws_N2j, -1)[...,None]
    #                 nus_N2j = np.mean(pnus_N2j, -2)
                
    #                 Ds_bst.append(np.mean(Ds_N2j * pnus_N2j, (-2,-1)).reshape(self.hs.shape))
    #                 ms_bst.append(np.mean(nus_N2j * mus_N2j, -1).reshape(self.hs.shape))
    #                 mus_bst.append(mus_N2j[...,0])
    #                 nus_bst.append(nus_N2j[...,0].reshape(self.hs.shape))

    #             elif self.fit.nu_method == 'knn':
    #                 hf = hf.astype(int)
    #                 h_max = np.max(hf)
    #                 proj_j = np.argpartition(Ds_N2j, hf, -1)[:,:h_max]
                    
    #                 Ds_bst.append((np.cumsum(np.mean(
    #                     Ds_N2j[np.repeat(np.arange(Xs_N2j.shape[0])[:,None], h_max, -1), proj_j], 0
    #                 ))[hf-1]/hf).reshape(self.hs.shape))
    #                 ms_bst.append((np.cumsum(np.mean(
    #                     mus_N2j[proj_j], 0
    #                 ))[hs-1]/hs).reshape(self.hs.shape))
    #                 mus_bst.append(mus_N2j[...,0])
    #                 nus_bst.append((np.cumsum(np.sum(
    #                     proj_j==0, 0
    #                 ))[hf-1]/hf).reshape(self.hs.shape))

    #             else:
    #                 raise('Only k-nearest-neighborhood (knn) and kernel smoothing (ksm) methods are supported now')
                
    #         Ds_bst = np.array(Ds_bst)
    #         ms_bst = np.array(ms_bst)
    #         mus_bst = np.array(mus_bst)
    #         nus_bst = np.array(nus_bst)
            
    #         offsets.append(np.mean(
    #             (mus_bst.reshape((n_sample,)+(1,)*self.hs.ndim) * nus_bst - ms_bst
    #             ).reshape((n_sample,)+(1,)*lamdas.ndim+self.hs.shape)
    #             * np.exp(- lamdas.reshape(lamdas.shape+(1,)*self.hs.ndim) 
    #                      * Ds_bst.reshape((n_sample,)+(1,)*lamdas.ndim+self.hs.shape)), 0
    #         ))
        
    #     return np.array(offsets)

    def get_phi(self):
        if self.offsets is None:
            offsets = 0
        else:
            offsets = self.offsets
            
        phis = (
            (self.xis.reshape((self.fit.data.n_node,)+(1,)*self.lamdas.ndim+self.hs.shape)
             - self.est())
            * np.exp(- self.lamdas.reshape(self.lamdas.shape+(1,)*self.hs.ndim) 
                     * self.Ds.reshape((self.fit.data.n_node,)+(1,)*self.lamdas.ndim+self.hs.shape))
            + offsets
        ) / np.sum(
            np.exp(- self.lamdas.reshape(self.lamdas.shape+(1,)*self.hs.ndim) 
                   * self.Ds.reshape((self.fit.data.n_node,)+(1,)*self.lamdas.ndim+self.hs.shape)), 0
        )

        return phis