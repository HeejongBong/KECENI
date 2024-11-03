import numpy as np
import numpy.random as random
import pandas as pd

from .KernelEstimate import KernelEstimate
from . import istarmap

class IT_broadcaster:
    def __init__(self, i0s, T0s):
        self.n_node = T0s.shape[-1]
        self.Tfs = T0s.reshape([-1, self.n_node])
        self.Txs = np.arange(self.Tfs.shape[0]).reshape(T0s.shape[:-1])
        self.b = np.broadcast(i0s, self.Txs)
        
    def __iter__(self):
        for i0, Tx in self.b:
            yield i0, self.Tfs[Tx]
        
class Fit:
    def __init__(self, model, data):
        self.data = data
        self.model = model

        self.mu_fit = self.model.mu_model.fit(data)
        self.pi_fit = self.model.pi_model.fit(data)
        self.cov_fit = self.model.cov_model.fit(data)
        
        self.xis = None
        
    def mu(self, T_N1, X_N2, G_N2):
        return self.mu_fit.predict(T_N1, X_N2, G_N2)

    def pi(self, T_N1, X_N2, G_N2):
        return self.pi_fit.predict(T_N1, X_N2, G_N2)

    def rT(self, n_sample, X_N2, G_N2):
        return self.pi_fit.sample(n_sample, X_N2, G_N2)
    
    def rX(self, n_sample, N2, G):
        return self.cov_fit.sample(n_sample, N2, G)
    
    def G_estimate(self, i0s, T0s, G0=None, n_X=1000, n_process=1, tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
                
        if G0 is None:
            G0 = self.data.G

        ITb = IT_broadcaster(i0s, T0s)
        
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.mu, map(
                lambda b_i: (b_i[1][None,G0.N1(b_i[0])], self.rX(n_X, G0.N2(b_i[0]), G0), G0.sub(G0.N2(b_i[0]))),
                ITb
            )),  total=ITb.b.size, leave=None, position=level_tqdm, desc='i0', smoothing=0))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.mu, map(
                    lambda b_i: (b_i[1][None,G0.N1(b_i[0])], self.rX(n_X, G0.N2(b_i[0]), G0), G0.sub(G0.N2(b_i[0]))),
                    ITb
                )), total=ITb.b.size, leave=None, position=level_tqdm, desc='i0', smoothing=0))
        
        return np.mean(r, -1).reshape(ITb.b.shape)

    def D_j(self, j, i0s, T0s=None, G0=None):
        if G0 is None:
            G0 = self.data.G
            if T0s is None:
                T0s = self.data.Ts
        elif T0s is None:
            raise
                
        T_N1j = self.data.Ts[self.data.G.N1(j)]

        ITb = IT_broadcaster(i0s, T0s)
        
        Ds_N2j = np.zeros(ITb.b.size)
        for ix, (i0, T0) in enumerate(ITb):
            T0_N1i0 = T0[G0.N1(i0)]
            Ds_N2j[ix] = self.model.delta(
                T_N1j, self.data.G.sub(self.data.G.N2(j)),
                T0_N1i0, G0.sub(G0.N2(i0))
            )

        return Ds_N2j.reshape(ITb.b.shape)
    
    def nuisance_j(self, j, n_X=100, seed=12345):
        np.random.seed(seed)
    
        T_N1j = self.data.Ts[self.data.G.N1(j)]
        
        if n_X > 0:
            Xs_N2j = np.concatenate([
                self.data.Xs[None,self.data.G.N2(j)],
                self.rX(
                    n_X, self.data.G.N2(j), self.data.G
                )
            ], 0)
        else:
            Xs_N2j = self.data.Xs[None,self.data.G.N2(j)]

        mus_N2j = self.mu_fit.predict(
            T_N1j, Xs_N2j, self.data.G.sub(self.data.G.N2(j))
        )
        pis_N2j = self.pi_fit.predict(
            T_N1j, Xs_N2j, self.data.G.sub(self.data.G.N2(j))
        )

        return mus_N2j[0], pis_N2j[0], np.mean(mus_N2j[1:]), np.mean(pis_N2j[1:])
    
    def set_xi(self, n_X=100, n_process=1, tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
        
        # def nuisance_j(self, j, n_X=100, seed=12345):
        
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.nuisance_j, map(
                lambda j: (j, n_X, np.random.randint(12345)),
                range(self.data.n_node)
            )), total=self.data.n_node, leave=None, position=level_tqdm, desc='xi_j', smoothing=0))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.nuisance_j, map(
                    lambda j: (j, n_X, np.random.randint(12345)),
                    range(self.data.n_node)
                )), total=self.data.n_node, leave=None, position=level_tqdm, desc='xi_j', smoothing=0))
                
        mus, pis, ms, vps = list(zip(*r))
        
        self.mus = np.array(mus)
        self.pis = np.array(pis)
        self.ms = np.array(ms)
        self.vps = np.array(vps)
        
        self.xis = (self.data.Ys - self.mus) * self.vps / self.pis + self.ms
        self.n_X = n_X
        
        return

    def kernel_AIPW(self, i0s, T0s=None, G0=None, 
                    lamdas=0, n_X=100, n_process=1, tqdm=None, level_tqdm=0):
        if G0 is None:
            G0 = self.data.G
            if T0s is None:
                T0s = self.data.Ts
        elif T0s is None:
            raise
            
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
            
        if self.xis is None:
            self.set_xi(n_X, n_process, tqdm, level_tqdm)
                
        lamdas = np.array(lamdas)

        # def D_j(self, j, i0s, T0s=None, G0=None):

        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.D_j, map(
                lambda j: (j, i0s, T0s, G0),
                range(self.data.n_node)
            )), total=self.data.n_node, leave=None, position=level_tqdm, desc='D_j', smoothing=0))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.D_j, map(
                    lambda j: (j, i0s, T0s, G0),
                    range(self.data.n_node)
                )), total=self.data.n_node, leave=None, position=level_tqdm, desc='D_j', smoothing=0))

        Ds = np.array(r)
        
        return KernelEstimate(self, i0s, T0s, G0, lamdas, Ds)
        
#         ws = np.exp(
#             - lamdas.reshape(lamdas.shape+(1,)*(Ds.ndim-1))
#             * Ds.reshape((self.data.n_node,)+(1,)*lamdas.ndim+Ds.shape[1:])
#         )
        
#         return np.sum(
#             self.xis.reshape((self.data.n_node,)+(1,)*(lamdas.ndim+Ds.ndim-1))
#             * ws, 0
#         ) / np.sum(ws, 0)

    def D_cv_j(self, j, i0s=None):
        nin = np.logical_not(np.isin(i0s, self.data.G.N2(j)))
        
        T_N1j = self.data.Ts[self.data.G.N1(j)]
        
        # def D_j(self, j, i0s, T0s=None, G0=None):

        D = np.full(i0s.shape, np.inf)
        D[...,nin] = self.D_j(j, i0s[nin])

        return D

    def loo_cv(self, lamdas, i0s=None, n_X=100, n_process=1, tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
            
        if i0s is None:
            i0s = np.arange(self.data.n_node)
            
        if self.xis is None:
            self.set_xi(n_X, n_process, tqdm, level_tqdm)
                
        lamdas = np.array(lamdas)

        # def D_cv_j(self, j, i0s):

        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.D_cv_j, map(
                lambda j: (j, i0s),
                range(self.data.n_node)
            )), total=self.data.n_node, leave=None, position=level_tqdm, desc='D_j', smoothing=0))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.D_cv_j, map(
                    lambda j: (j, i0s),
                    range(self.data.n_node)
                )), total=self.data.n_node, leave=None, position=level_tqdm, desc='D_j', smoothing=0))

        Ds = np.array(r)
        
        return KernelEstimate(self, i0s, self.data.Ts, self.data.G, lamdas, Ds)
                
#         ws = np.exp(
#             - lamdas.reshape(lamdas.shape+(1,)*(Ds.ndim-1))
#             * Ds.reshape((self.data.n_node,)+(1,)*lamdas.ndim+Ds.shape[1:])
#         )
        
#         return self.xis[i0s], np.sum(
#             self.xis.reshape((self.data.n_node,)+(1,)*(lamdas.ndim+Ds.ndim-1))
#             * ws, 0
#         ) / np.sum(ws, 0)

                        
        