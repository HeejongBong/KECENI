import numpy as np
import numpy.random as random
import pandas as pd

from . import istarmap
from .Data import Data
from .KernelEstimate import KernelEstimate, CrossValidationEstimate
from .IT_broadcaster import IT_broadcaster

        
class Fit:
    def __init__(self, model, data=None, n_X=100, js=None, 
                 mus=None, pis=None, ms=None, vps=None,
                 n_process=None, tqdm=None, level_tqdm=0):
        if data is not None and js is None:
            js = np.arange(data.n_node)
            
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
        
        self.model = model
        self.data = data

        self.mu_fit = self.model.mu_model.fit(data)
        self.pi_fit = self.model.pi_model.fit(data)
        self.cov_fit = self.model.cov_model.fit(data)
        
        self.n_X = n_X
        self.js = js
        
        if data is not None:
            if (mus is None or pis is None or ms is None or vps is None):
                # def xi_j(self, j, n_X=100, seed=12345):
                if n_process == 1:
                    from itertools import starmap
                    r = list(tqdm(starmap(self.nu_j, map(
                        lambda j: (j, n_X, np.random.randint(12345)), js
                    )), total=len(js), leave=None, position=level_tqdm, desc='j', smoothing=0))

                elif n_process is None or n_process > 1:
                    from multiprocessing import Pool
                    with Pool(n_process) as p:   
                        r = list(tqdm(p.istarmap(self.nu_j, map(
                            lambda j: (j, n_X, np.random.randint(12345)), js
                        )), total=len(js), leave=None, position=level_tqdm, desc='j', smoothing=0))

                self.mus, self.pis, self.ms, self.vps = list(zip(*r))

                self.mus = np.array(self.mus)
                self.pis = np.array(self.pis)
                self.ms = np.array(self.ms)
                self.vps = np.array(self.vps)

            else:
                self.mus = np.array(mus)
                self.pis = np.array(pis)
                self.ms = np.array(ms)
                self.vps = np.array(vps)
        
            self.xis = (self.data.Ys[js] - self.mus) * self.vps / self.pis + self.ms
        
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
        
        elif n_process is None or n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.mu, map(
                    lambda b_i: (b_i[1][None,G0.N1(b_i[0])], self.rX(n_X, G0.N2(b_i[0]), G0), G0.sub(G0.N2(b_i[0]))),
                    ITb
                )), total=ITb.b.size, leave=None, position=level_tqdm, desc='i0', smoothing=0))
        
        return np.mean(r, -1).reshape(ITb.b.shape)
    
    def nu_j(self, j, n_X=100, seed=12345):
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

        mus_N2j = self.mu(
            T_N1j, Xs_N2j, self.data.G.sub(self.data.G.N2(j))
        )
        pis_N2j = self.pi(
            T_N1j, Xs_N2j, self.data.G.sub(self.data.G.N2(j))
        )
        
        return mus_N2j[0], pis_N2j[0], np.mean(mus_N2j[1:]), np.mean(pis_N2j[1:])
    
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

    def kernel_AIPW(self, i0s, T0s=None, G0=None,
                    n_process=None, tqdm=None, level_tqdm=0):
        if G0 is None:
            G0 = self.data.G
            if T0s is None:
                T0s = self.data.Ts
        elif T0s is None:
            raise
            
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable

        # def D_j(self, j, i0s, T0s=None, G0=None):

        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.D_j, map(
                lambda j: (j, i0s, T0s, G0), self.js
            )), total=len(self.js), leave=None, position=level_tqdm, desc='j', smoothing=0))
        
        elif n_process is None or n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.D_j, map(
                    lambda j: (j, i0s, T0s, G0), self.js
                )), total=len(self.js), leave=None, position=level_tqdm, desc='j', smoothing=0))

        Ds = np.array(r)
        
        return KernelEstimate(self, i0s, T0s, G0, Ds)
    
    def D_cv_j(self, j, i0s):
        G0 = self.data.G
        T0s = self.data.Ts
        nin = np.logical_not(np.isin(i0s, self.data.G.N2(j)))
        
        T_N1j = self.data.Ts[self.data.G.N1(j)]
            
        ITb = IT_broadcaster(i0s[nin], T0s)
        
        Ds_N2j = np.zeros(ITb.b.size)
        for ix, (i0, T0) in enumerate(ITb):
            T0_N1i0 = T0[G0.N1(i0)]
            Ds_N2j[ix] = self.model.delta(
                T_N1j, self.data.G.sub(self.data.G.N2(j)),
                T0_N1i0, G0.sub(G0.N2(i0))
            )
                
        D = np.full(i0s.shape, np.inf)
        D[...,nin] = Ds_N2j

        return D

    def cv(self, i0s=None, n_process=None, tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
            
        if i0s is None:
            i0s = np.arange(self.data.n_node)
                
        # def D_cv_j(self, j, i0s):

        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.D_cv_j, map(
                lambda j: (j, i0s), self.js
            )), total=len(self.js), leave=None, position=level_tqdm, desc='j', smoothing=0))
        
        elif n_process is None or n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.D_cv_j, map(
                    lambda j: (j, i0s), self.js
                )), total=len(self.js), leave=None, position=level_tqdm, desc='j', smoothing=0))

        Ds_cv = np.array(r)
        
        return CrossValidationEstimate(
            self, i0s, self.data.Ts, self.data.G, Ds_cv
        )
    
    def cov_bst(self, n_bst): 
        return [
            self.model.cov_model.fit(
                Data(None, None, Xs_i, self.data.G)
            )
            for Xs_i in self.rX(n_bst, np.arange(self.data.n_node), self.data.G)
        ]

    def sub(self, ind_js):
        # def __init__(self, model, data=None, n_X=100, js=None, 
        #          mus=None, pis=None, ms=None, vps=None,
        #          n_process=None, tqdm=None, level_tqdm=0):
        
        return Fit(self.model, self.data, self.n_X, self.js[ind_js],
                   self.mus[ind_js], self.pis[ind_js], self.ms[ind_js], self.vps[ind_js])
        
                        
        