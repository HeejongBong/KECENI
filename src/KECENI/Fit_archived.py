import numpy as np
import numpy.random as random
import pandas as pd
from tqdm.notebook import tqdm

import KECENI.Model

class Fit:
    def __init__(self, data, mu_model, pi_model, cov_model, delta):
        self.data = data

        self.mu_model = mu_model
        self.pi_model = pi_model
        self.cov_model = cov_model

        self.mu_fit = self.mu_model.fit(self)
        self.pi_fit = self.pi_model.fit(self)
        self.cov_fit = self.cov_model.fit(self)

        self.delta = delta
        
    def mu(self, T_N1, X_N2, G_N2):
        return self.mu_fit.predict(T_N1, X_N2, G_N2)

    def pi(self, T_N1, X_N2, G_N2):
        return self.pi_fit.predict(T_N1, X_N2, G_N2)
    
    def rX(self, n_sample, N2, G):
        return self.cov_fit.sample(n_sample, N2, G)

    def G_estimate(self, i0, T0, G0=None, n_sample=1000, return_std=False):
        if G0 is None:
            G0 = self.data.G

        n_node0 = G0.shape[0]
        N1i0 = np.concatenate([[i0], np.nonzero(G0[i0])[0]])
        N2i0 = pd.unique(np.concatenate([
            np.concatenate([[j], np.nonzero(G0[j])[0]])
            for j in N1i0
        ]))
        T0_N1i0 = T0[N1i0]
        
        T0s_N1i0 = np.repeat(T0_N1i0[None,:], n_sample, 0)
        Xs_N2i0 = self.rX(n_sample, N2i0, G0)
        mus_N2i0 = self.mu(T0s_N1i0, Xs_N2i0, G0[np.ix_(N2i0, N2i0)])
        
        if return_std:
            return np.mean(mus_N2i0), np.std(mus_N2i)/np.sqrt(n_sample)
        else:
            return np.mean(mus_N2i0)

    def smoothed_G_estimate(self, i0, T0, G0=None, 
                    n_sample=1000, lamdas=1, n_process=1, leave_tqdm=True):
        lamdas = np.array(lamdas)
        
        if G0 is None:
            G0 = self.G
        n_node0 = G0.shape[0]
        N1i0 = np.concatenate([[i0], np.nonzero(G0[i0])[0]])
        N2i0 = pd.unique(np.concatenate([
            np.concatenate([[j], np.nonzero(G0[j])[0]])
            for j in N1i0
        ]))
        
        T0_N1i0 = T0[N1i0]
        Xs_N2i0 = self.rX(n_sample, N2i0, G0)
        G0_N2i0 = G0[np.ix_(N2i0,N2i0)]
        if n_process == 1:
            from itertools import starmap
            r = list(starmap(self.m_D, tqdm(
                ((j, T0_N1i0, Xs_N2i0, G0_N2i0, lamdas, n_sample) 
                 for j in range(self.n_node)),
                total=self.n_node, leave=leave_tqdm, desc='j'
            )))
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(p.starmap(self.m_D, tqdm(
                    ((j, T0_N1i0, Xs_N2i0, G0_N2i0, lamdas, n_sample) 
                     for j in range(self.n_node)),
                    total=self.n_node, leave=leave_tqdm, desc='j'
                )))

        ms = np.array(list(r))[...,0].T
        Ds = np.array(list(r))[...,1].T
        
        return np.sum(ms * np.exp(- lamdas[...,None] * Ds), -1) \
               / np.sum(np.exp(- lamdas[...,None] * Ds), -1)

    def m_D(self, j, T_N1k, Xs_N2k, G_N2k, lamdas=1, n_sample=100):
        lamdas = np.array(lamdas)
        
        N1j = self.N1s[j]
        N2j = self.N2s[j]
        
        Y_j = self.Ys[j]
        T_N1j = self.Ts[N1j]
        X_N2j = self.Xs[N2j]
        G_N2j = self.G[np.ix_(N2j,N2j)]
    
        Xs_N2j = np.concatenate([
            X_N2j[None,...], self.rX(n_sample-1, N2j, self.G)
        ], 0)
        Ts_N1j = np.repeat(T_N1j[None,:], Xs_N2j.shape[0], 0)
        Ts_N1k = np.repeat(T_N1k[None,:], Xs_N2k.shape[0], 0)
        
        Ds_N2j = self.delta(Ts_N1k, Xs_N2k, G_N2k, Ts_N1j, Xs_N2j, G_N2j)
        Ws_N2j = np.exp(- lamdas[...,None,None] * Ds_N2j)
        pnus_N2j = Ws_N2j / np.mean(Ws_N2j, -1)[...,None]
        nus_N2j = np.mean(pnus_N2j, -2)
    
        m_j = np.mean(nus_N2j * self.mu(Ts_N1j, Xs_N2j, G_N2j), -1)
        D = np.mean(Ds_N2j * pnus_N2j, (-2,-1))
        return m_j, D

    def DR_estimate(self, i0, T0, G0=None, 
                    n_sample=1000, lamdas=1, n_process=1, leave_tqdm=True):
        lamdas = np.array(lamdas)
        
        if G0 is None:
            G0 = self.G
        n_node0 = G0.shape[0]
        N1i0 = np.concatenate([[i0], np.nonzero(G0[i0])[0]])
        N2i0 = pd.unique(np.concatenate([
            np.concatenate([[j], np.nonzero(G0[j])[0]])
            for j in N1i0
        ]))
        
        T0_N1i0 = T0[N1i0]
        Xs_N2i0 = self.rX(n_sample, N2i0, G0)
        G0_N2i0 = G0[np.ix_(N2i0,N2i0)]
        if n_process == 1:
            from itertools import starmap
            r = list(starmap(self.xi_D, tqdm(
                ((j, T0_N1i0, Xs_N2i0, G0_N2i0, lamdas, n_sample) 
                 for j in range(self.n_node)),
                total=self.n_node, leave=leave_tqdm, desc='j'
            )))
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(p.starmap(self.xi_D, tqdm(
                    ((j, T0_N1i0, Xs_N2i0, G0_N2i0, lamdas, n_sample) 
                     for j in range(self.n_node)),
                    total=self.n_node, leave=leave_tqdm, desc='j'
                )))

        xis = np.array(r).T[...,0,:]
        Ds = np.array(r).T[...,1,:]
        
        return np.sum(xis * np.exp(- lamdas[...,None] * Ds), -1) \
               / np.sum(np.exp(- lamdas[...,None] * Ds), -1)

    def xi_D(self, j, T_N1k, Xs_N2k, G_N2k, lamdas=1, n_sample=100):
        lamdas = np.array(lamdas)
        
        N1j = self.N1s[j]
        N2j = self.N2s[j]
        
        Y_j = self.Ys[j]
        T_N1j = self.Ts[N1j]
        X_N2j = self.Xs[N2j]
        G_N2j = self.G[np.ix_(N2j,N2j)]
    
        Xs_N2j = np.concatenate([
            X_N2j[None,...], self.rX(n_sample-1, N2j, self.G)
        ], 0)
        Ts_N1j = np.repeat(T_N1j[None,:], Xs_N2j.shape[0], 0)
        Ts_N1k = np.repeat(T_N1k[None,:], Xs_N2k.shape[0], 0)
        
        Ds_N2j = self.delta(Ts_N1k, Xs_N2k, G_N2k, Ts_N1j, Xs_N2j, G_N2j)
        Ws_N2j = np.exp(- lamdas[...,None,None] * Ds_N2j)
        pnus_N2j = Ws_N2j / np.mean(Ws_N2j, -1)[...,None]
        nus_N2j = np.mean(pnus_N2j, -2)
    
        varpi_j = np.mean(self.pi(Ts_N1j, Xs_N2j, G_N2j))
        m_j = np.mean(nus_N2j * self.mu(Ts_N1j, Xs_N2j, G_N2j), -1)
    
        xi = (Y_j - self.mu(T_N1j, Xs_N2j[0], G_N2j)) \
              * varpi_j/self.pi(T_N1j, Xs_N2j[0], G_N2j)*nus_N2j[...,0] \
              + m_j
        D = np.mean(Ds_N2j * pnus_N2j, (-2,-1))
        return xi, D

    def loo_cv(self, lamdas, n_cv=100, n_sample=100, n_process=1):
        lamdas = np.array(lamdas)

        ks_cv = random.choice(np.arange(self.n_node), n_cv)
        Ys_cv = np.zeros(lamdas.shape + (n_cv,))

        for iter_k in tqdm(range(n_cv), desc='k'):
            k = ks_cv[iter_k]
            N1k = self.N1s[k]
            N2k = self.N2s[k]
            
            mk = np.delete(np.arange(self.n_node), self.N2s[k])
            
            Ys_mk = self.Ys[mk]
            Ts_mk = self.Ts[mk]
            Xs_mk = self.Xs[mk]
            G_mk = self.G[np.ix_(mk,mk)]

            model_mk = KECENI.Model(
                self.mu_model, self.pi_model, self.cov_model, self.delta
            )
            fit_mk = model_mk.fit(Ys_mk, Ts_mk, Xs_mk, G_mk)
        
            Ys_cv[:,iter_k] = fit_mk.DR_estimate(
                k, self.Ts, self.G, n_sample=n_sample, 
                lamdas=lamdas, n_process=n_process, leave_tqdm = False
            )

        return np.mean((Ys_cv - self.Ys[ks_cv])**2, -1)