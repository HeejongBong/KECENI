import numpy as np
import numpy.random as random
import pandas as pd

from .KernelEstimate import KernelEstimate
from . import istarmap
        
class Fit:
    def __init__(self, model, data, nu_method='ksm'):
        self.data = data
        self.model = model

        self.mu_fit = self.model.mu_model.fit(data)
        self.pi_fit = self.model.pi_model.fit(data)
        self.cov_fit = self.model.cov_model.fit(data)

        self.nu_method = nu_method
        
    def mu(self, T_N1, X_N2, G_N2):
        return self.mu_fit.predict(T_N1, X_N2, G_N2)

    def pi(self, T_N1, X_N2, G_N2):
        return self.pi_fit.predict(T_N1, X_N2, G_N2)

    def rT(self, n_sample, X_N2, G_N2):
        return self.pi_fit.sample(n_sample, X_N2, G_N2)
    
    def rX(self, n_sample, N2, G):
        return self.cov_fit.sample(n_sample, N2, G)
    
    def G_estimate(self, i0, T0, G0=None, n_sample=1000, return_std=False):
        if G0 is None:
            G0 = self.data.G

        N1i0 = G0.N1(i0)
        N2i0 = G0.N2(i0)
        
        Xs_N2i0 = self.rX(n_sample, N2i0, G0)
        mus_N2i0 = self.mu(T0[None,N1i0], Xs_N2i0, G0.sub(N2i0))
        
        if return_std:
            return np.mean(mus_N2i0), np.std(mus_N2i0)/np.sqrt(n_sample)
        else:
            return np.mean(mus_N2i0)

    def EIF_j(self, j, i0, T0, G0, lamdas=1, hs=1, n_T=100, n_X=110, n_X0=None, seed=12345):
        np.random.seed(seed)

        if n_X0 is None:
            n_X0 = n_X
        
        lamdas = np.array(lamdas)
        hs = np.array(hs)
        hf = hs.flatten()
        
        T0_N1i0 = T0[G0.N1(i0)]
        Xs_N2i0 = self.rX(
            n_X, G0.N2(i0), G0
        )

        if n_T > 0:
            Ts_N1j = np.concatenate([
                self.data.Ts[None,self.data.G.N1(j)],
                self.rT(1, self.rX(n_T, self.data.G.N2(j), self.data.G), 
                        self.data.G.sub(self.data.G.N2(j)))[0]
            ], 0)
        else:
            Ts_N1j = self.data.Ts[None,self.data.G.N1(j)]

        if n_X > 0:
            Xs_N2j = np.concatenate([
                self.data.Xs[None,self.data.G.N2(j)],
                self.rX(
                    n_X, self.data.G.N2(j), self.data.G
                )
            ], 0)
        else:
            Xs_N2j = self.data.Xs[None,None,self.data.G.N2(j)]
        
        mus_N2j = self.mu(
            Ts_N1j[:,None], Xs_N2j, self.data.G.sub(self.data.G.N2(j))
        )
        pis_N2j = self.pi(
            Ts_N1j[0,None], Xs_N2j, self.data.G.sub(self.data.G.N2(j))
        )
        # pis_N2j = self.pi(
        #     Ts_N1j[0,None], Xs_N2j[0], self.data.G.sub(self.data.G.N2(j))
        # )
        Ds_N2j = self.model.delta(
            Ts_N1j[:,None,None],
            Xs_N2j[None,None,:], self.data.G.sub(self.data.G.N2(j)),
            T0_N1i0[None,None,None],
            Xs_N2i0[None,:,None], G0.sub(G0.N2(i0))
        )
        # Ds_N2j = self.model.delta(
        #     Ts_N1j[:,None,None],
        #     Xs_N2j[:,None,:], self.data.G.sub(self.data.G.N2(j)),
        #     T0_N1i0[None,None,None],
        #     Xs_N2i0[:,:,None], G0.sub(G0.N2(i0))
        # )
        
        if self.nu_method == 'knn':
            proj_j = np.argpartition(Ds_N2j, hf, -1)[...,:h_max]
            index_arr = list(np.ix_(*[range(i) for i in Ds_N2j.shape]))
            index_arr[-1] = proj_j
            
            Ds_bst = np.cumsum(np.mean(Ds_N2j[*index_arr], -2), -1)[...,hf-1]/hf
            ms_bst = np.cumsum(np.mean(
                mus_N2j[np.arange(n_T+1)[:,None,None], proj_j], -2
            ), -1)[...,hs-1]/hs
            mus_bst = mus_N2j[...,0]
            nus_bst = np.cumsum(np.sum(proj_j==0, -2), -1)[...,hf-1]/hf

            D = Ds_bst[0].reshape(hs.shape)

            if self.data.Ys is None:
                xi = ms_bst[0].reshape(hs.shape)
            else:
                xi = (
                    (Ys[j] - mus_bst[0]) \
                    * np.mean(pis_N2j) / pis_N2j[0] \
                    * nus_bst[0] \
                    + ms_bst[0]
                ).reshape(hs.shape)

            if n_T > 0:
                offset = np.mean(
                    (mus_bst[...,1:] * nus_bst[...,1:] - ms_bst[...,1:])
                    * np.exp(- lamdas.reshape(lamdas.shape+(1,1)) 
                             * Ds_bst[...,1:].reshape((1,)*lamdas.ndim+(len(hf),n_T))), -1
                ).reshape(lamdas.shape+hs.shape)
            else:
                offset = np.zeros(lamdas.shape+hs.shape)
            
        if self.nu_method == 'ksm':
            Ws_N2j = np.exp(- hf[...,None,None,None] 
                            * (Ds_N2j - np.min(Ds_N2j, -1)[...,None]))
            pnus_N2j = Ws_N2j / np.mean(Ws_N2j, -1)[...,None]
            nus_N2j = np.mean(pnus_N2j, -2)            

            Ds_bst = np.mean(Ds_N2j * pnus_N2j, (-2,-1))
            ms_bst = np.mean(nus_N2j * mus_N2j, -1)
            mus_bst = mus_N2j[...,0]
            nus_bst = nus_N2j[...,0]

            D = Ds_bst[...,0].reshape(hs.shape)
            
            if self.data.Ys is None:
                xi = ms_bst[...,0].reshape(hs.shape)
            else:
                xi = (
                    (self.data.Ys[j] - mus_bst[0]) 
                    * np.mean(pis_N2j) / pis_N2j[0] 
                    * nus_bst[...,0]
                    + ms_bst[...,0]
                ).reshape(hs.shape)

            wm = np.mean(
                    np.exp(- lamdas.reshape(lamdas.shape+(1,1)) 
                           * Ds_bst.reshape((1,)*lamdas.ndim+(len(hf),n_T+1))), -1
                ).reshape(lamdas.shape+hs.shape)
            
            if n_T > 0:
                offset = np.mean(
                    (mus_bst[...,1:] * nus_bst[...,1:] - ms_bst[...,1:])
                    * np.exp(- lamdas.reshape(lamdas.shape+(1,1)) 
                             * Ds_bst[...,1:].reshape((1,)*lamdas.ndim+(len(hf),n_T))), -1
                ).reshape(lamdas.shape+hs.shape)
            else:
                offset = np.zeros(lamdas.shape+hs.shape)
            
        else:
            raise('Only knearest neighborhood (knn) and kernel smoothing (ksm) methods are supported now')
        
        return D, xi, wm, offset

    def kernel_EIF(self, i0, T0, G0=None, 
                   lamdas=1, hs=1, n_T=100, n_X=110, n_X0=None, n_process=1,
                   tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
                
        lamdas = np.array(lamdas)
        hs = np.array(hs)

        if G0 is None:
            G0 = self.data.G

        # EIF_j(self, j, i0, T0, G0, lamdas=1, hs=1, n_T=100, n_X=110, n_X0=None, seed=12345)
        
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.EIF_j,
                (
                    (j, i0, T0, G0, lamdas, hs, n_T, n_X, n_X0, np.random.randint(12345))
                    for j in range(self.data.n_node)
                )
            ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.EIF_j,
                    (
                        (j, i0, T0, G0, lamdas, hs, n_T, n_X, n_X0, np.random.randint(12345))
                        for j in range(self.data.n_node)
                    )
                ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0))

        Ds, xis, wms, offsets = list(zip(*r))

        return KernelEstimate(self, i0, T0, G0, lamdas, hs, 
                              np.array(Ds), np.array(xis), np.array(wms), np.array(offsets))

    def average_EIF(self, T0, G0=None, lamdas=1, hs=1, n_T=0, n_X=100,
                    n_process=1, tqdm=None, level_tqdm=0):
        if G0 is None:
            G0 = self.data.G

        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable

        # kernel_EIF(self, i0, T0, G0=None, 
        #            lamdas=1, hs=1, n_T=100, n_X=110, n_X0=None, n_process=1,
        #            tqdm=None, level_tqdm=0)
        
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.kernel_EIF,
                ((i0, T0, G0, lamdas, hs, n_T, n_X, None, 1, tqdm, level_tqdm+1) 
                 for i0 in range(G0.n_node))
            ), total=self.data.n_node, leave=None, position=level_tqdm, desc='i0', smoothing=0))
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.kernel_EIF,
                    ((i0, T0, G0, lamdas, hs, n_T, n_X, None, 1, None, level_tqdm+1) 
                     for i0 in range(G0.n_node))
                ), total=self.data.n_node, leave=None, position=level_tqdm, desc='i0', smoothing=0))
        
        return r

    def loo_cv(self, lamdas, hs, n_cv=100, n_sample=100, n_process=1, 
               tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable

        lamdas = np.array(lamdas)
        hs = np.array(hs)

        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.mu_pi_j,
                map(lambda j: (j, n_T, n_X), range(n_node))
            ), total=n_node, leave=None, position=0, desc='j', smoothing=0))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.mu_pi_j,
                    map(lambda j: (j, n_T, n_X), range(n_node))
                ), total=n_node, leave=None, position=0, desc='j', smoothing=0))

        mus, pis = np.swapaxes(np.array(r), 1, 0)
        
        ms = np.mean(mus, -1)
        varpis = np.mean(pis, -1)
                        
        ks_cv = random.choice(np.arange(self.data.n_node), n_cv, replace=False)

        # def lKo_cv(self, K, lamdas, hs, n_sample, mus, pis, ms, varpis,
        #            tqdm = None, level_tqdm=0):
        
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.lKo_cv,
                (([k], lamdas, hs, n_sample, mus, pis, ms, varpis,
                  tqdm, level_tqdm+1) 
                 for k in ks_cv)
            ), total=len(ks_cv), leave=None, position=level_tqdm, desc='i_cv', smoothing=0))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.lKo_cv,
                    (([k], lamdas, hs, n_sample, mus, pis, ms, varpis,
                      None, level_tqdm+1) 
                     for k in ks_cv)
                ), total=len(ks_cv), leave=None, position=level_tqdm, desc='i_cv', smoothing=0))      

        return np.array([r_i[0][0] for r_i in r]), np.array([r_i[1][0] for r_i in r])
    
    def lko_cv(self, lamdas, hs, n_cv=100, n_sample=100, n_leave=100, n_trial=10, n_process=1, 
               tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable

        Xs_G = np.concatenate([
            self.data.Xs[None,...], self.rX(n_sample-1, np.arange(self.data.n_node), self.data.G)
        ], 0)

        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.mu,
                [(self.data.Ts[None,self.data.G.N1(j)],
                  # Xs_G[:,self.data.G.N2(j)], 
                  np.concatenate([
                      self.data.Xs[None,self.data.G.N2(j),:], self.rX(n_sample-1, self.data.G.N2(j), self.data.G)
                  ], 0), 
                  self.data.G.sub(self.data.G.N2(j)))
                 for j in np.arange(self.data.n_node)]
            ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.mu,
                    [(self.data.Ts[None,self.data.G.N1(j)],
                      # Xs_G[:,self.data.G.N2(j)], 
                      np.concatenate([
                          self.data.Xs[None,self.data.G.N2(j),:], self.rX(n_sample-1, self.data.G.N2(j), self.data.G)
                      ], 0), 
                      self.data.G.sub(self.data.G.N2(j)))
                     for j in np.arange(self.data.n_node)]
                ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0)) 

        mus = np.array(r)
        
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.pi,
                [(self.data.Ts[None,self.data.G.N1(j)],
                  # Xs_G[:,self.data.G.N2(j)], 
                  np.concatenate([
                      self.data.Xs[None,self.data.G.N2(j),:], self.rX(n_sample-1, self.data.G.N2(j), self.data.G)
                  ], 0),
                  self.data.G.sub(self.data.G.N2(j)))
                 for j in np.arange(self.data.n_node)]
            ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.pi,
                    [(self.data.Ts[None,self.data.G.N1(j)],
                      # Xs_G[:,self.data.G.N2(j)], 
                      np.concatenate([
                          self.data.Xs[None,self.data.G.N2(j),:], self.rX(n_sample-1, self.data.G.N2(j), self.data.G)
                      ], 0),
                      self.data.G.sub(self.data.G.N2(j)))
                     for j in np.arange(self.data.n_node)]
                ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0)) 

        pis = np.array(r)
        
        ms = np.mean(mus, -1)
        varpis = np.mean(pis, -1)
        
        Ks=[]
        for i_cv in np.arange(n_cv):
            
            for i_trial in np.arange(n_trial):
                K = random.choice(np.arange(self.data.n_node), n_leave, replace=False)
                N2K = pd.unique(np.concatenate([
                    self.data.G.N2(k) for k in K
                ]))
        
                if len(N2K) < self.data.n_node - n_leave:
                    break
                elif i_trial == n_trial-1:
                    raise
                    
            Ks.append(K)
            
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.lKo_cv,
                ((K, lamdas, hs, n_sample, mus, pis, ms, varpis,
                  tqdm, level_tqdm+1) 
                 for K in Ks)
            ), total=n_cv, leave=None, position=level_tqdm, desc='i_cv', smoothing=0))

        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.lKo_cv,
                    ((K, lamdas, hs, n_sample, mus, pis, ms, varpis,
                      None, level_tqdm+1) 
                     for K in Ks)
                ), total=n_cv, leave=None, position=level_tqdm, desc='i_cv', smoothing=0))      

        return np.array([r_i[0] for r_i in r]), np.array([r_i[1] for r_i in r])
    
    def lKo_cv(self, K, lamdas, hs, n_sample, mus, pis, ms, varpis,
                   tqdm = None, level_tqdm=0):
        hs = np.array(hs)
        h_shape = hs.shape
        hs = hs.flatten()
        
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable

        if len(K) > 1:
            tqdm_k = tqdm
        else:
            def tqdm_k(iterable, *args, **kwargs):
                return iterable
                
        # n_sample = Xs_G.shape[0]
        
        N2K = pd.unique(np.concatenate([
                    self.data.G.N2(k) for k in K
                ]))
        VmK = np.delete(np.arange(self.data.n_node), N2K)
        
        xis = np.zeros(len(K))
        xhs = np.zeros((len(K),)+lamdas.shape+h_shape)
        for i_k, k in tqdm_k(enumerate(K), total=len(K), leave=None, position=level_tqdm, desc='k'):
            T_N1k = self.data.Ts[self.data.G.N1(k)]
            Xs_N2k = np.concatenate([
                self.data.Xs[None,self.data.G.N2(k),:], self.rX(n_sample-1, self.data.G.N2(k), self.data.G)
            ], 0)
            G_N2k = self.data.G.sub(self.data.G.N2(k))

            xis[i_k] = (self.data.Ys[k] - mus[k,0]) * varpis[k] / pis[k,0] + ms[k]

            ds = np.zeros((len(VmK), n_sample, n_sample))
            for i_j, j in tqdm(enumerate(VmK), total=len(VmK), leave=None, position=level_tqdm+1, desc='j'):
                T_N1j = self.data.Ts[self.data.G.N1(j)]
                Xs_N2j = np.concatenate([
                    self.data.Xs[None,self.data.G.N2(j),:], self.rX(n_sample-1, self.data.G.N2(j), self.data.G)
                ], 0)
                G_N2j = self.data.G.sub(self.data.G.N2(j))
                ds[i_j] = self.model.delta(T_N1k[None,None], Xs_N2k[:,None], G_N2k,
                                           T_N1j[None,None], Xs_N2j[None,:], G_N2j)

            if self.nu_method == 'ksm':
                Ws = np.exp(- hs[...,None,None,None] 
                              * (ds - np.min(ds, -1)[...,None]))
                pnus = Ws / np.mean(Ws, -1)[...,None]
                nus = np.mean(pnus, -2)

                mns = np.mean(nus * mus[VmK], -1)
                xns = ((self.data.Ys[VmK] - mus[VmK, 0]) 
                       * varpis[VmK] / pis[VmK,0] * nus[...,0] + mns)
                Ds = np.mean(ds * pnus, (-2, -1))

            elif self.nu_method == 'knn':
                hs = hs.astype(int)
                h_max = np.max(hs)

                proj = (np.argpartition(ds, hs, -1)[...,:h_max]).transpose((2,0,1))

                mns = np.cumsum(np.mean(
                    mus[VmK[:,None],proj], -1
                ), 0)[hs-1]/hs[:,None]
                xns = ((self.data.Ys[VmK] - mus[VmK, 0]) 
                       * varpis[VmK] / pis[VmK,0] 
                       * (np.cumsum(np.sum(proj==0, -1), 0)[hs-1]/hs[:,None])
                       + mns)
                Ds = np.cumsum(np.mean(
                    ds[np.arange(len(VmK))[:,None], 
                       np.arange(n_sample), proj], -1
                ), 0)[hs-1]/hs[:,None]

            else:
                raise('Only k-nearest-neighborhood (knn) and kernel smoothing (ksm) methods are supported now')

            xhs[i_k] = (np.sum(
                xns
                * np.exp(- lamdas.reshape(lamdas.shape+(1,)*(hs.ndim+1)) 
                         * Ds), -1
            ) / np.sum(
                np.exp(- lamdas.reshape(lamdas.shape+(1,)*(hs.ndim+1)) 
                       * Ds), -1
            )).reshape(lamdas.shape+h_shape)

        return xis, xhs
    
    def mu_pi_j(self, j, n_T, n_X):
        if n_T > 0:
            Ts_N1j = np.concatenate([
                self.data.Ts[None,self.data.G.N1(j)],
                self.rT(1, self.rX(n_T, self.data.G.N2(j), self.data.G), 
                        self.data.G.sub(self.data.G.N2(j)))[0]
            ], 0)
        else:
            Ts_N1j = self.data.Ts[None,self.data.G.N1(j)]

        if n_X > 0:
            Xs_N2j = np.concatenate([
                self.data.Xs[None,self.data.G.N2(j)],
                self.rX(
                    n_X, self.data.G.N2(j), self.data.G
                )
            ], 0)
        else:
            Xs_N2j = self.data.Xs[None,None,self.data.G.N2(j)]
        
        return (
            self.mu(Ts_N1j[:,None], Xs_N2j, self.data.G.sub(self.data.G.N2(j))),
            self.pi(Ts_N1j[0,None], Xs_N2j, self.data.G.sub(self.data.G.N2(j)))
        )
    
class ITX_broadcaster:
    def __init__(self, i0s, T0s, Xs_N2i0s):
        self.n_node = T0s.shape[-1]
        self.ifs = i0s.flatten()
        self.Tfs = T0s.reshape((-1, self.n_node))       
        self.Xfs = Xs_N2i0s.flatten()
        
        self.ixs = np.arange(self.ifs.shape[0]).reshape(i0s.shape)
        self.Txs = np.arange(self.Tfs.shape[0]).reshape(T0s.shape[:-1])
        self.b = np.broadcast(self.ixs, self.Txs)
        
    def __iter__(self):
        for ix, Tx in self.b:
            yield self.ifs[ix], self.Tfs[Tx], self.Xfs[ix]