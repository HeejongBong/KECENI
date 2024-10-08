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

    def AIPW_j(self, j, i0s, T0s, G0, lamdas=0, hs=0, n_T=100, n_X=110, n_X0=None, seed=12345):
        np.random.seed(seed)
    
        lamdas = np.array(lamdas)
        if self.nu_method == 'fixed':
            hs = np.array(0)
        else:
            hs = np.array(hs)
        hf = hs.flatten()

        if n_X0 is None:
            n_X0 = n_X
        
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
            Xs_N2j = self.data.Xs[None,self.data.G.N2(j)]
        
        mus_N2j = self.mu(
            Ts_N1j[:,None], Xs_N2j, self.data.G.sub(self.data.G.N2(j))
        )
        pis_N2j = self.pi(
            Ts_N1j[0,None], Xs_N2j, self.data.G.sub(self.data.G.N2(j))
        )

        ITb = IT_broadcaster(i0s, T0s)

        if self.nu_method == 'fixed':
            Ds_N2j = np.zeros((ITb.b.size, n_T+1))
            for ix, (i0, T0) in enumerate(ITb):
                T0_N1i0 = T0[G0.N1(i0)]
                Ds_N2j[ix] = self.model.delta(
                    Ts_N1j, None, self.data.G.sub(self.data.G.N2(j)),
                    T0_N1i0, None, G0.sub(G0.N2(i0))
                )
        else:
            Ds_N2j = np.zeros((ITb.b.size, n_T+1, n_X0, n_X+1))
            for ix, (i0, T0) in enumerate(ITb):
                T0_N1i0 = T0[G0.N1(i0)]
                Xs_N2i0 = self.rX(n_X0, G0.N2(i0), G0)
                Ds_N2j[ix] = self.model.delta(
                    Ts_N1j[:,None,None],
                    Xs_N2j[None,None,:], self.data.G.sub(self.data.G.N2(j)),
                    T0_N1i0[None,None,None],
                    Xs_N2i0[None,:,None], G0.sub(G0.N2(i0))
                )

        if self.nu_method == 'fixed':
            Ds_bst = Ds_N2j
            mus_bst = mus_N2j[...,0]
            nus_bst = np.full((ITb.b.size,) + mus_bst.shape, 1)
            mns_bst = np.mean(mus_N2j, -1) * nus_bst

        elif self.nu_method == 'ksm':
            Ws_N2j = np.exp(- hf.reshape((-1,)+(1,)*4)
                              * (Ds_N2j - np.min(Ds_N2j, -1)[...,None]))
            pnus_N2j = Ws_N2j / np.mean(Ws_N2j, -1)[...,None]
            nus_N2j = np.mean(pnus_N2j, -2)

            Ds_bst = np.mean(Ds_N2j * pnus_N2j, (-2,-1))
            mns_bst = np.mean(nus_N2j * mus_N2j, -1)
            mus_bst = mus_N2j[...,0]
            nus_bst = nus_N2j[...,0]
            
        elif self.nu_method == 'knn':
            hf = hf.astype(int)
            h_max = np.max(hf)

            proj_j = np.argpartition(Ds_N2j, hf, -1)[...,:h_max]
            index_arr = list(np.ix_(*[range(i) for i in Ds_N2j.shape]))
            index_arr[-1] = proj_j

            Ds_bst = np.moveaxis(np.cumsum(np.mean(Ds_N2j[*index_arr], -2), -1)[...,hf-1]/hf, -1, 0)
            mns_bst = np.moveaxis(np.cumsum(np.mean(
                mus_N2j[np.arange(n_T+1)[:,None,None], proj_j], -2
            ), -1)[...,hf-1]/hf, -1, 0)
            mus_bst = mus_N2j[...,0]
            nus_bst = np.moveaxis(np.cumsum(np.sum(proj_j==0, -2), -1)[...,hf-1]/hf, -1, 0)
        
        else:
            raise('Only knearest neighborhood (knn) and kernel smoothing (ksm) methods are supported now')

        D = Ds_bst[...,0].reshape(hs.shape + ITb.b.shape)

        if self.data.Ys is None:
            xi = mns_bst[...,0].reshape(hs.shape + ITb.b.shape)
        else:
            xi = (
                (self.data.Ys[j] - mus_bst[0]) 
                * np.mean(pis_N2j) / pis_N2j[0] 
                * nus_bst[...,0]
                + mns_bst[...,0]
            ).reshape(hs.shape + ITb.b.shape)

        ws = np.exp(
            - lamdas.reshape(lamdas.shape+(1,)*3) 
            * Ds_bst.reshape((1,)*lamdas.ndim+(len(hf),ITb.b.size,n_T+1))
        )

        if n_T > 0:
            wm = np.mean(ws, -1).reshape(lamdas.shape+hs.shape+ITb.b.shape)
            wxm = np.mean(ws * mns_bst, -1).reshape(lamdas.shape+hs.shape+ITb.b.shape)
        else:
            wm = None
            wxm = None

        return D, xi, wm, wxm

    def kernel_AIPW(self, i0s, T0s=None, G0=None, 
                    lamdas=0, hs=0, n_T=100, n_X=110, n_X0=None, 
                    n_process=1, tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
                
        lamdas = np.array(lamdas)
        if self.nu_method == 'fixed':
            hs = np.array(0)
        else:
            hs = np.array(hs)

        if G0 is None:
            G0 = self.data.G
            if T0s is None:
                T0s = self.data.Ts
        elif T0s is None:
            raise

        if n_X0 is None:
            n_X0 = n_X

        # AIPW_j(self, j, i0s, T0s, G0, lamdas=1, hs=1, n_T=100, n_X=110, n_X0=None, seed=12345)
        
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.AIPW_j, map(
                lambda j: (j, i0s, T0s, G0, lamdas, hs, n_T, n_X, n_X0, np.random.randint(12345)),
                range(self.data.n_node)
            )), total=self.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.AIPW_j, map(
                    lambda j: (j, i0s, T0s, G0, lamdas, hs, n_T, n_X, n_X0, np.random.randint(12345)),
                    range(self.data.n_node)
                )), total=self.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0))

        Ds, xis, wms, wxms = list(zip(*r))

        return KernelEstimate(self, i0s, T0s, G0, lamdas, hs, 
                              np.array(Ds), np.array(xis), np.array(wms), np.array(wxms))

    def loo_cv_j(self, j, lamdas, hs=0, i0s=None, n_X=110, n_X0=None, seed=12345):
        np.random.seed(seed)
    
        lamdas = np.array(lamdas)
        if self.nu_method == 'fixed':
            hs = np.array(0)
        else:
            hs = np.array(hs)
        hf = hs.flatten()

        if i0s is None:
            i0s = np.arange(self.data.n_node)

        T0 = self.data.Ts
        G0 = self.data.G

        nin = np.logical_not(np.isin(i0s, self.data.G.N2(j)))

        if n_X0 is None:
            n_X0 = n_X
        
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
            T_N1j[None], Xs_N2j, self.data.G.sub(self.data.G.N2(j))
        )
        pis_N2j = self.pi(
            T_N1j[None], Xs_N2j, self.data.G.sub(self.data.G.N2(j))
        )
        xi = (self.data.Ys[j] - mus_N2j[0]) * np.mean(pis_N2j) / pis_N2j[0] + np.mean(mus_N2j)

        if self.nu_method == 'fixed':
            Ds_N2j = np.zeros(i0s[nin].shape)
            for ix, i0 in enumerate(i0s[nin]):
                T0_N1i0 = T0[G0.N1(i0)]
                Ds_N2j[ix] = self.model.delta(
                    T_N1j, None, self.data.G.sub(self.data.G.N2(j)),
                    T0_N1i0, None, G0.sub(G0.N2(i0))
                )
        else:
            Ds_N2j = np.zeros(i0s[nin].shape + (n_X0, n_X+1))
            for ix, i0 in enumerate(i0s[nin]):
                T0_N1i0 = T0[G0.N1(i0)]
                Xs_N2i0 = self.rX(n_X0, G0.N2(i0), G0)
                Ds_N2j[ix] = self.model.delta(
                    T_N1j[None,None],
                    Xs_N2j[None,:], self.data.G.sub(self.data.G.N2(j)),
                    T0_N1i0[None,None],
                    Xs_N2i0[:,None], G0.sub(G0.N2(i0))
                )

        if self.nu_method == 'fixed':
            Ds_bst = Ds_N2j
            mus_bst = mus_N2j[...,0]
            nus_bst = np.full(i0s[nin].shape + mus_bst.shape, 1)
            mns_bst = np.mean(mus_N2j, -1) * nus_bst

        elif self.nu_method == 'ksm':
            Ws_N2j = np.exp(- hf.reshape((-1,)+(1,)*3)
                            * (Ds_N2j - np.min(Ds_N2j, -1)[...,None]))
            pnus_N2j = Ws_N2j / np.mean(Ws_N2j, -1)[...,None]
            nus_N2j = np.mean(pnus_N2j, -2)

            Ds_bst = np.mean(Ds_N2j * pnus_N2j, (-2,-1))
            mns_bst = np.mean(nus_N2j * mus_N2j, -1)
            mus_bst = mus_N2j[...,0]
            nus_bst = nus_N2j[...,0]
            
        elif self.nu_method == 'knn':
            hf = hf.astype(int)
            h_max = np.max(hf)

            proj_j = np.argpartition(Ds_N2j, hf, -1)[...,:h_max]
            index_arr = list(np.ix_(*[range(i) for i in Ds_N2j.shape]))
            index_arr[-1] = proj_j

            Ds_bst = np.moveaxis(np.cumsum(np.mean(Ds_N2j[*index_arr], -2), -1)[...,hf-1]/hf, -1, 0)
            mns_bst = np.moveaxis(np.cumsum(np.mean(
                mus_N2j[np.arange(n_T+1)[:,None,None], proj_j], -2
            ), -1)[...,hf-1]/hf, -1, 0)
            mus_bst = mus_N2j[...,0]
            nus_bst = np.moveaxis(np.cumsum(np.sum(proj_j==0, -2), -1)[...,hf-1]/hf, -1, 0)
        
        else:
            raise('Only knearest neighborhood (knn) and kernel smoothing (ksm) methods are supported now')

        D = np.full(hs.shape + i0s.shape, np.inf)
        D[...,nin] = Ds_bst.reshape(hs.shape + (-1,))

        xn = np.zeros(hs.shape + i0s.shape)
        xn[...,nin] = (
            (self.data.Ys[j] - mus_bst) 
            * np.mean(pis_N2j) / pis_N2j[0] 
            * nus_bst
            + mns_bst
        )

        return xi, D, xn

    def loo_cv(self, lamdas, hs=0, i0s=None, n_X=100, n_X0=None, n_process=1, tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable

        lamdas = np.array(lamdas)
        if self.nu_method == 'fixed':
            hs = np.array(0)
        else:
            hs = np.array(hs)

        if i0s is None:
            i0s = np.arange(self.data.n_node)

        if n_X0 is None:
            n_X0 = n_X
                        
        # def loo_cv_j(self, j, lamdas, hs, i0s, n_X=110, n_X0=None, seed=12345):
        
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.loo_cv_j, map(
                lambda j: (j, lamdas, hs, i0s, n_X, n_X0, np.random.randint(12345)),
                range(self.data.n_node)
            )), total=self.data.n_node, leave=None, position=level_tqdm, desc='i_cv', smoothing=0))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.loo_cv_j, map(
                    lambda j: (j, lamdas, hs, i0s, n_X, n_X0, np.random.randint(12345)),
                    range(self.data.n_node)
                )), total=self.data.n_node, leave=None, position=level_tqdm, desc='i_cv', smoothing=0))

        xis, Ds, xns = list(zip(*r))

        xis = np.array(xis)[i0s]
        Ds = np.array(Ds); xns = np.array(xns)

        xhs = np.sum(
            xns
            * np.exp(- lamdas.reshape(lamdas.shape+(1,)*(Ds.ndim)) 
                     * Ds), -Ds.ndim
        ) / np.sum(
            np.exp(- lamdas.reshape(lamdas.shape+(1,)*(Ds.ndim)) 
                   * Ds), -Ds.ndim
        )

        return xis, xhs