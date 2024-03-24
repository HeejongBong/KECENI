import numpy as np
import numpy.random as random
import pandas as pd

import KECENI

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
    def __init__(self, fit, i0, T0, G0, lamdas, hs, Ds, xis, mode):
        self.fit = fit
        
        self.i0 = i0
        self.T0 = T0
        self.G0 = G0

        self.lamdas = np.array(lamdas)
        self.hs = np.array(hs)
        
        self.Ds = Ds
        self.xis = xis
        self.mode = mode

        self.offsets = None

    def est(self, lamdas=None, offset=False, n_sample=100, hac_kernel=parzen_kernel, 
            id_bst=None, tqdm=None, level_tqdm=0, **kwargs):
        if id_bst is None:
            id_bst = np.arange(self.fit.data.G.n_node)

        if offset:
            if lamdas is None:
                lamdas = self.lamdas
                if self.offsets is None:
                    self.offsets = self.get_offset(lamdas, n_sample, tqdm, level_tqdm)
                offsets = self.offsets
            else:
                lamdas = np.array(lamdas)
                offsets = self.get_offset(lamdas, n_sample, tqdm, level_tqdm)
                
            phis = (
                self.xis[id_bst].reshape((len(id_bst),)+(1,)*lamdas.ndim+self.hs.shape)
                * np.exp(- lamdas.reshape(lamdas.shape+(1,)*self.hs.ndim) 
                         * self.Ds[id_bst].reshape((len(id_bst),)+(1,)*lamdas.ndim+self.hs.shape))
                + offsets[id_bst]
            ) / np.sum(
                np.exp(- lamdas.reshape(lamdas.shape+(1,)*self.hs.ndim) 
                       * self.Ds[id_bst].reshape((len(id_bst),)+(1,)*lamdas.ndim+self.hs.shape)), 0
            )

            return np.sum(phis, 0)
        else:
            if lamdas is None:
                lamdas = self.lamdas
            else:
                lamdas = np.array(lamdas)
                        
            return np.sum(
                self.xis[id_bst].reshape((len(id_bst),)+(1,)*lamdas.ndim+self.hs.shape)
                * np.exp(- lamdas.reshape(lamdas.shape+(1,)*self.hs.ndim) 
                         * self.Ds[id_bst].reshape((len(id_bst),)+(1,)*lamdas.ndim+self.hs.shape)), 0
            ) / np.sum(
                np.exp(- lamdas.reshape(lamdas.shape+(1,)*self.hs.ndim) 
                       * self.Ds[id_bst].reshape((len(id_bst),)+(1,)*lamdas.ndim+self.hs.shape)), 0
            )

    def mse(self, lamdas=None, n_sample=100, hac_kernel=parzen_kernel, 
            id_bst=None, tqdm=None, level_tqdm=0, abs=False, **kwargs):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
        
        if lamdas is None:
            lamdas = self.lamdas
            if self.offsets is None:
                self.offsets = self.get_offset(lamdas, n_sample, tqdm, level_tqdm)
            offsets = self.offsets
        else:
            lamdas = np.array(lamdas)
            offsets = self.get_offset(lamdas, n_sample, tqdm, level_tqdm)

        if id_bst is None:
            id_bst = np.arange(self.fit.data.G.n_node)
            
        phis = (
            (self.xis[id_bst].reshape((len(id_bst),)+(1,)*lamdas.ndim+self.hs.shape)
             - self.est(llamdas=lamdas, id_bst=id_bst, offset=True))
            * np.exp(- lamdas.reshape(lamdas.shape+(1,)*self.hs.ndim) 
                     * self.Ds[id_bst].reshape((len(id_bst),)+(1,)*lamdas.ndim+self.hs.shape))
            + offsets[id_bst]
        ) / np.sum(
            np.exp(- lamdas.reshape(lamdas.shape+(1,)*self.hs.ndim) 
                   * self.Ds[id_bst].reshape((len(id_bst),)+(1,)*lamdas.ndim+self.hs.shape)), 0
        )

        if abs:
            return np.sum(np.abs(phis) * np.tensordot(
                hac_kernel(self.fit.data.G.dist, **kwargs), np.abs(phis), axes=(-1,0)
            ), -phis.ndim)
        else:
            return np.sum(phis * np.tensordot(
                hac_kernel(self.fit.data.G.dist, **kwargs), phis, axes=(-1,0)
            ), -phis.ndim)

    def ste(self, lamdas=None, abs=False, n_sample=100, hac_kernel=parzen_kernel, 
            id_bst=None, tqdm=None, level_tqdm=0, **kwargs):
        return np.sqrt(self.mse(lamdas=lamdas, abs=abs, n_sample=n_sample, hac_kernel=hac_kernel, 
                                id_bst=id_bst, tqdm=tqdm, level_tqdm=level_tqdm, **kwargs))

    def get_offset(self, lamdas=None, n_sample=100, tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable

        if lamdas is None:
            lamdas = self.lamdas
                
        hf = self.hs.flatten()

        T0_N1i0 = self.T0[self.G0.N1(self.i0)]
        G0_N2i0 = self.G0.sub(self.G0.N2(self.i0))

        offsets = list()
        for j in tqdm(range(self.fit.data.n_node), smoothing=0, desc='j', leave=None, position=level_tqdm):
            Xs_N2j = np.concatenate([
                self.fit.data.Xs[None,self.fit.data.G.N2(j)],
                self.fit.rX(n_sample-1, self.fit.data.G.N2(j), self.fit.data.G)
            ], 0)
            Ts_N1j = self.fit.rT(1, Xs_N2j, self.fit.data.G.sub(self.fit.data.G.N2(j)))[0]
            
            Ds_bst = list()
            ms_bst = list()
            mus_bst = list()
            nus_bst = list()
            
            for T_N1j in Ts_N1j:
                Xs_N2j = np.concatenate([
                    self.fit.data.Xs[None,self.fit.data.G.N2(j)],
                    self.fit.rX(n_sample-1, self.fit.data.G.N2(j), self.fit.data.G)
                ], 0)
                Xs_N2i0 = self.fit.rX(n_sample, self.G0.N2(self.i0), self.G0)
        
                Ds_N2j = self.fit.model.delta(
                    np.repeat(T0_N1i0[None,:], n_sample, 0), Xs_N2i0, 
                    self.G0.sub(self.G0.N2(self.i0)),
                    np.repeat(T_N1j[None,:], n_sample, 0), Xs_N2j, 
                    self.fit.data.G.sub(self.fit.data.G.N2(j))
                )
                
                if self.fit.nu_method == 'ksm':
                    Ws_N2j = np.exp(- hf[...,None,None] 
                                    * (Ds_N2j - np.min(Ds_N2j, -1)[...,None]))
                    pnus_N2j = Ws_N2j / np.mean(Ws_N2j, -1)[...,None]
                    nus_N2j = np.mean(pnus_N2j, -2)
                    mus_N2j = self.fit.mu(np.repeat(T_N1j[None,:], n_sample, 0), Xs_N2j, 
                                          self.fit.data.G.sub(self.fit.data.G.N2(j)))
                
                    Ds_bst.append(np.mean(Ds_N2j * pnus_N2j, (-2,-1)).reshape(self.hs.shape))
                    ms_bst.append(np.mean(nus_N2j * mus_N2j, -1).reshape(self.hs.shape))
                    mus_bst.append(mus_N2j[...,0])
                    nus_bst.append(nus_N2j[...,0].reshape(self.hs.shape))

                elif self.fit.nu_method == 'knn':
                    Ws_N2j = np.exp(- hf[...,None,None] 
                                    * (Ds_N2j - np.min(Ds_N2j, -1)[...,None]))
                    pnus_N2j = Ws_N2j / np.mean(Ws_N2j, -1)[...,None]
                    nus_N2j = np.mean(pnus_N2j, -2)
                    mus_N2j = self.fit.mu(np.repeat(T_N1j[None,:], n_sample, 0), Xs_N2j, 
                                          self.fit.data.G.sub(self.fit.data.G.N2(j)))
                
                    Ds_bst.append((np.cumsum(np.mean(
                        Ds_N2j[np.repeat(np.arange(Xs_N2j.shape[0])[:,None], h_max, -1), proj_j], 0
                    ))[hf-1]/hf).reshape(self.hs.shape))
                    ms_bst.append((np.cumsum(np.mean(
                        self.mu(Ts_N1j, Xs_N2j, G_N2j)[proj_j], 0
                    ))[hs-1]/hs).reshape(self.hs.shape))
                    mus_bst.append(mus_N2j[...,0])
                    nus_bst.append((np.cumsum(np.sum(
                        proj_j==0, 0
                    ))[hf-1]/hf).reshape(self.hs.shape))

                else:
                    raise('Only k-nearest-neighborhood (knn) and kernel smoothing (ksm) methods are supported now')
                
            Ds_bst = np.array(Ds_bst)
            ms_bst = np.array(ms_bst)
            mus_bst = np.array(mus_bst)
            nus_bst = np.array(nus_bst)
            
            offsets.append(np.mean(
                (mus_bst.reshape((n_sample,)+(1,)*self.hs.ndim) * nus_bst - ms_bst
                ).reshape((n_sample,)+(1,)*lamdas.ndim+self.hs.shape)
                * np.exp(- lamdas.reshape(lamdas.shape+(1,)*self.hs.ndim) 
                         * Ds_bst.reshape((n_sample,)+(1,)*lamdas.ndim+self.hs.shape)), 0
            ))
        
        return np.array(offsets)

    # def calibrate_bw(self, bws, lamdas=None, abs=False, hac_kernel=parzen_kernel, 
    #                  n_bst=1000, n_id1=None, return_ss=False, tqdm=None, level_tqdm=0, **kwargs):
    #     if lamdas is None:
    #         lamdas = self.lamdas
    #     else:
    #         lamdas = np.array(lamdas)

    #     if n_id1 is None:
    #         n_id1 = self.fit.data.G.n_node * 0.15

    #     if tqdm is None:
    #         def tqdm(iterable, *args, **kwargs):
    #             return iterable

    #     ss1_bst = np.zeros((n_bst,)+lamdas.shape+self.hs.shape)
    #     ss2_bst = np.zeros((n_bst,)+bws.shape+lamdas.shape+self.hs.shape)

    #     for i_bst in tqdm(range(n_bst), total=n_bst, leave=None, position=level_tqdm, desc='i_bst', smoothing=0):
    #         id1_bst = {np.random.choice(np.arange(self.fit.data.G.n_node))}
    #         for k in np.arange(n_id1-1):
    #             N1_id1 = set(np.concatenate([self.fit.data.G.N1(i) for i in id1_bst]))
    #             dN1_id1 = N1_id1.difference(id1_bst)
    #             if len(dN1_id1) == 0:
    #                 id1_bst.add(np.random.choice(
    #                     np.delete(np.arange(self.fit.data.G.n_node), list(id1_bst))
    #                 ))
    #             else:
    #                 id1_bst.add(np.random.choice(list(dN1_id1)))
    #         N2_id1 = set(np.concatenate([self.fit.data.G.N2(i) for i in id1_bst]))
    #         id2_bst = set(range(self.fit.data.G.n_node)).difference(
    #             set(np.concatenate([self.fit.data.G.N2(i) for i in N2_id1]))
    #         )
            
    #         id1_bst = np.array(list(id1_bst))
    #         id2_bst = np.array(list(id2_bst))
            
    #         ss1_bst[i_bst] = (self.est(lamdas=lamdas, id_bst=id1_bst)-self.est(lamdas=lamdas, id_bst=id2_bst))**2
    #         ss2_bst[i_bst] = np.array([
    #             self.mse(lamdas=lamdas, abs=abs, hac_kernel=hac_kernel, id_bst=id1_bst, bw=bw_i, **kwargs) 
    #             + self.mse(lamdas=lamdas, abs=abs, hac_kernel=hac_kernel, id_bst=id2_bst, bw=bw_i, **kwargs)
    #             for bw_i in bws
    #         ])

    #     if return_ss:
    #         return ss1_bst, ss2_bst
    #     else:
    #         return np.mean(ss1_bst[:,None]/ss2_bst, 0)
        
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

    def dissimilarity(self, Y_j, T_N1j, Xs_N2j, G_N2j, T_N1k, Xs_N2k, G_N2k, hs=1, mode=0):
        hs = np.array(hs)
        hf = hs.flatten()
        
        Ts_N1j = np.repeat(T_N1j[None,:], Xs_N2j.shape[0], 0)
        Ts_N1k = np.repeat(T_N1k[None,:], Xs_N2k.shape[0], 0)
        Ds_N2j = self.model.delta(Ts_N1k, Xs_N2k, G_N2k, Ts_N1j, Xs_N2j, G_N2j)

        if self.nu_method == 'knn':
            hf = hf.astype(int)
            h_max = np.max(hf)
            
            proj_j = np.argpartition(Ds_N2j, hf, -1)[:,:h_max]
            D = np.cumsum(np.mean(
                Ds_N2j[np.repeat(np.arange(Xs_N2j.shape[0])[:,None], h_max, -1), proj_j], 0
            ))[hf-1]/hf

            if mode == 0:
                return D.reshape(hs.shape)
        
            varpi_j = np.mean(self.pi(Ts_N1j, Xs_N2j, G_N2j))
            m_j = np.cumsum(np.mean(
                self.mu(Ts_N1j, Xs_N2j, G_N2j)[proj_j], 0
            ))[hs-1]/hs

            if mode == 1:
                return D.reshape(hs.shape), m_j.reshape(hs.shape)
        
            xi = (Y_j - self.mu(T_N1j, Xs_N2j[0], G_N2j)) \
               * varpi_j/self.pi(T_N1j, Xs_N2j[0], G_N2j) \
               * np.cumsum(np.sum(proj_j==0, 0))[hf-1]/hf \
               + m_j
            
        elif self.nu_method == 'ksm':
            Ws_N2j = np.exp(- hf[...,None,None] 
                            * Ds_N2j)
                            # * (Ds_N2j - np.min(Ds_N2j, -1)[...,None]))
            pnus_N2j = Ws_N2j / np.mean(Ws_N2j, -1)[...,None]
            nus_N2j = np.mean(pnus_N2j, -2)
            D = np.mean(Ds_N2j * pnus_N2j, (-2,-1))

            if mode == 0:
                return D.reshape(hs.shape)
        
            varpi_j = np.mean(self.pi(Ts_N1j, Xs_N2j, G_N2j))
            m_j = np.mean(nus_N2j * self.mu(Ts_N1j, Xs_N2j, G_N2j), -1)

            if mode == 1:
                return D.reshape(hs.shape), m_j.reshape(hs.shape)
        
            xi = (Y_j - self.mu(T_N1j, Xs_N2j[0], G_N2j)) \
               * varpi_j/self.pi(T_N1j, Xs_N2j[0], G_N2j) \
               * nus_N2j[...,0] \
               + m_j
            
        else:
            raise('Only k-nearest-neighborhood (knn) and kernel smoothing (ksm) methods are supported now')
        
        return D.reshape(hs.shape), xi.reshape(hs.shape)

    def G_estimate(self, i0, T0, G0=None, n_sample=1000, return_std=False):
        if G0 is None:
            G0 = self.data.G

        N1i0 = G0.N1(i0)
        N2i0 = G0.N2(i0)
        
        T0s_N1i0 = np.repeat(T0[None,N1i0], n_sample, 0)
        Xs_N2i0 = self.rX(n_sample, N2i0, G0)
        mus_N2i0 = self.mu(T0s_N1i0, Xs_N2i0, G0.sub(N2i0))
        
        if return_std:
            return np.mean(mus_N2i0), np.std(mus_N2i0)/np.sqrt(n_sample)
        else:
            return np.mean(mus_N2i0)

    def kernel_AIPW(self, i0, T0, Xs_N2i0=None, G0=None, 
                    lamdas=1, hs=1, n_sample=1000, n_process=1, mode=2,
                    tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
                
        lamdas = np.array(lamdas)
        hs = np.array(hs)

        if G0 is None:
            G0 = self.data.G
        
        N1i0 = G0.N1(i0)
        N2i0 = G0.N2(i0)

        T0_N1i0 = T0[N1i0]
        G0_N2i0 = G0.sub(N2i0)
        
        if Xs_N2i0 is None:
            Xs_N2i0 = self.rX(n_sample, N2i0, G0)
        
        # Xs_G = np.concatenate([
        #     self.data.Xs[None,...], self.rX(n_sample-1, np.arange(self.data.n_node), self.data.G)
        # ], 0)

        # dissimilarity(self, Y_j, T_N1j, Xs_N2j, G_N2j, T_N1k, Xs_N2k, G_N2k, hs=1, mode=0)
        
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.dissimilarity,
                ((self.data.Ys[j], self.data.Ts[self.data.G.N1(j)],
                  # Xs_G[:,self.data.G.N2(j)], 
                  np.concatenate([
                      self.data.Xs[None,self.data.G.N2(j),:], self.rX(n_sample-1, self.data.G.N2(j), self.data.G)
                  ], 0),
                  self.data.G.sub(self.data.G.N2(j)),
                  T0_N1i0, Xs_N2i0, G0_N2i0, hs, mode)
                 for j in range(self.data.n_node))
            ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.dissimilarity, 
                    ((self.data.Ys[j], self.data.Ts[self.data.G.N1(j)],
                      # Xs_G[:,self.data.G.N2(j)], 
                      np.concatenate([
                          self.data.Xs[None,self.data.G.N2(j),:], self.rX(n_sample-1, self.data.G.N2(j), self.data.G)
                      ], 0),
                      self.data.G.sub(self.data.G.N2(j)),
                      T0_N1i0, Xs_N2i0, G0_N2i0, hs, mode) 
                     for j in range(self.data.n_node))
                ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0))

        Ds = np.array(r)[:,0,...]
        xis = np.array(r)[:,1,...]

        return KernelEstimate(self, i0, T0, G0, lamdas, hs, Ds, xis, mode)

    def DR_estimate(self, i0, T0, Xs_N2i0=None, G0=None, 
                    lamdas=1, hs=1, n_sample=1000, n_process=1, mode=2,
                    return_std=False, hac_kernel = parzen_kernel, 
                    tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
                
        lamdas = np.array(lamdas)
        hs = np.array(hs)

        if G0 is None:
            G0 = self.data.G
        
        N1i0 = G0.N1(i0)
        N2i0 = G0.N2(i0)

        T0_N1i0 = T0[N1i0]
        G0_N2i0 = G0.sub(N2i0)
        
        if Xs_N2i0 is None:
            Xs_N2i0 = self.rX(n_sample, N2i0, G0)
        
        # Xs_G = np.concatenate([
        #     self.data.Xs[None,...], self.rX(n_sample-1, np.arange(self.data.n_node), self.data.G)
        # ], 0)

        # dissimilarity(self, Y_j, T_N1j, Xs_N2j, G_N2j, T_N1k, Xs_N2k, G_N2k, hs=1, mode=0)
        
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.dissimilarity,
                ((self.data.Ys[j], self.data.Ts[self.data.G.N1(j)],
                  # Xs_G[:,self.data.G.N2(j)], 
                  np.concatenate([
                      self.data.Xs[None,self.data.G.N2(j),:], self.rX(n_sample-1, self.data.G.N2(j), self.data.G)
                  ], 0),
                  self.data.G.sub(self.data.G.N2(j)),
                  T0_N1i0, Xs_N2i0, G0_N2i0, hs, mode)
                 for j in range(self.data.n_node))
            ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.dissimilarity, 
                    ((self.data.Ys[j], self.data.Ts[self.data.G.N1(j)],
                      # Xs_G[:,self.data.G.N2(j)], 
                      np.concatenate([
                          self.data.Xs[None,self.data.G.N2(j),:], self.rX(n_sample-1, self.data.G.N2(j), self.data.G)
                      ], 0),
                      self.data.G.sub(self.data.G.N2(j)),
                      T0_N1i0, Xs_N2i0, G0_N2i0, hs, mode) 
                     for j in range(self.data.n_node))
                ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0))

        Ds = np.array(r)[:,0,...]
        xis = np.array(r)[:,1,...]
        
        psi = np.sum(
            xis.reshape((self.data.n_node,)+(1,)*lamdas.ndim+hs.shape)
            * np.exp(- lamdas.reshape(lamdas.shape+(1,)*hs.ndim) 
                     * Ds.reshape((self.data.n_node,)+(1,)*lamdas.ndim+hs.shape)), 0
        ) / np.sum(
            np.exp(- lamdas.reshape(lamdas.shape+(1,)*hs.ndim) 
                   * Ds.reshape((self.data.n_node,)+(1,)*lamdas.ndim+hs.shape)), 0
        )

        if return_std:
            phis = (
                (xis.reshape((self.data.n_node,)+(1,)*lamdas.ndim+hs.shape)
                 - psi)
                * np.exp(- lamdas.reshape(lamdas.shape+(1,)*hs.ndim) 
                         * Ds.reshape((self.data.n_node,)+(1,)*lamdas.ndim+hs.shape))
            ) / np.sum(
                np.exp(- lamdas.reshape(lamdas.shape+(1,)*hs.ndim) 
                       * Ds.reshape((self.data.n_node,)+(1,)*lamdas.ndim+hs.shape)), 0
            )
            return psi, np.sqrt(
                np.abs(phis).T[...,None,:] @ hac_kernel(self.data.G.dist(), G=self.data.G) 
                @ np.abs(phis).T[...,:,None]
            ).T[0,0]
        else:
            return psi

    def DR_average(self, T0, lamdas=1, hs=1, 
                   n_process=1, tqdm=None, level_tqdm=0):

        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable

#         def DR_estimate(self, i0, T0, Xs_N2i0=None, G0=None, 
#                         lamdas=1, hs=1, n_sample=1000, n_process=1, mode=2, 
#                         tqdm=None, level_tqdm=0):
        
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.DR_estimate,
                ((i0, T0, self.data.Xs[None,self.data.G.N2(i0)], self.data.G,
                  lamdas, hs, 1, 1, 2, tqdm, level_tqdm+1) 
                 for i0 in range(self.data.n_node))
            ), total=self.data.n_node, leave=None, position=level_tqdm, desc='i0', smoothing=0))
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.DR_estimate,
                ((i0, T0, self.data.Xs[None,self.data.G.N2(i0)], self.data.G,
                  lamdas, hs, 1, 1, 2, None, level_tqdm+1) 
                 for i0 in range(self.data.n_node))
            ), total=self.data.n_node, leave=None, position=level_tqdm, desc='i0', smoothing=0))
        
        return np.mean(np.array(r), 0)

    def loo_cv(self, lamdas, hs, n_cv=100, n_sample=100, n_process=1, 
               tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable

        lamdas = np.array(lamdas)
        hs = np.array(hs)

        # Xs_G = np.concatenate([
        #     self.data.Xs[None,...], self.rX(n_sample-1, np.arange(self.data.n_node), self.data.G)
        # ], 0)

        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.mu,
                [(np.repeat(self.data.Ts[None,self.data.G.N1(j)], n_sample, 0),
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
                    [(np.repeat(self.data.Ts[None,self.data.G.N1(j)], n_sample, 0),
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
                [(np.repeat(self.data.Ts[None,self.data.G.N1(j)], n_sample, 0),
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
                    [(np.repeat(self.data.Ts[None,self.data.G.N1(j)], n_sample, 0),
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
                        
        ks_cv = random.choice(np.arange(self.data.n_node), n_cv, replace=False)

        # def lKo_cv(self, K, lamdas, hs, n_sample, mus, pis, ms, varpis,
        #            tqdm = None, level_tqdm=0):
        
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.lKo_cv,
                (([k], lamdas, hs, n_sample, mus, pis, ms, varpis,
                  tqdm, level_tqdm+1) 
                 for k in ks_cv)
            ), total=len(ks_cv), leave=None, position=level_tqdm, desc='k', smoothing=0))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.lKo_cv,
                    (([k], lamdas, hs, n_sample, mus, pis, ms, varpis,
                      None, level_tqdm+1) 
                     for k in ks_cv)
                ), total=len(ks_cv), leave=None, position=level_tqdm, desc='k', smoothing=0))      

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
                [(np.repeat(self.data.Ts[None,self.data.G.N1(j)], n_sample, 0),
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
                    [(np.repeat(self.data.Ts[None,self.data.G.N1(j)], n_sample, 0),
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
                [(np.repeat(self.data.Ts[None,self.data.G.N1(j)], n_sample, 0),
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
                    [(np.repeat(self.data.Ts[None,self.data.G.N1(j)], n_sample, 0),
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
                
        # n_sample = Xs_G.shape[0]
        
        N2K = pd.unique(np.concatenate([
                    self.data.G.N2(k) for k in K
                ]))
        VmK = np.delete(np.arange(self.data.n_node), N2K)
        
        xis = np.zeros(len(K))
        xhs = np.zeros((len(K),)+lamdas.shape+h_shape)
        for i_k, k in tqdm(enumerate(K), leave=None, position=level_tqdm, desc='k'):
            Ts_N1k = np.repeat(self.data.Ts[None,self.data.G.N1(k)], n_sample, 0)
            Xs_N2k = np.concatenate([
                self.data.Xs[None,self.data.G.N2(k),:], self.rX(n_sample-1, self.data.G.N2(k), self.data.G)
            ], 0)
            G_N2k = self.data.G.sub(self.data.G.N2(k))

            xis[i_k] = (self.data.Ys[k] - mus[k,0]) * varpis[k] / pis[k,0] + ms[k]

            ds = np.zeros((len(VmK), n_sample, n_sample))
            for i_j, j in tqdm(enumerate(VmK), leave=None, position=level_tqdm+1, desc='j'):
                Ts_N1j = np.repeat(self.data.Ts[None,self.data.G.N1(j)], n_sample, 0)
                Xs_N2j = np.concatenate([
                    self.data.Xs[None,self.data.G.N2(j),:], self.rX(n_sample-1, self.data.G.N2(j), self.data.G)
                ], 0)
                G_N2j = self.data.G.sub(self.data.G.N2(j))
                ds[i_j] = self.model.delta(Ts_N1k, Xs_N2k, G_N2k, Ts_N1j, Xs_N2j, G_N2j)

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