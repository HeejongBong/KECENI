import numpy as np
import numpy.random as random
import pandas as pd

import KECENI

class Fit:
    def __init__(self, data, mu_model, pi_model, cov_model, delta, 
                 nu_method='knn', n_sample=1000, n_process=1, tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
                
        self.data = data

        self.mu_model = mu_model
        self.pi_model = pi_model
        self.cov_model = cov_model

        self.mu_fit = self.mu_model.fit(data)
        self.pi_fit = self.pi_model.fit(data)
        self.cov_fit = self.cov_model.fit(data)

        self.delta = delta
        self.nu_method = nu_method

        self.n_sample = n_sample
        self.Xs_G = np.concatenate([
            self.data.Xs[None,...], self.rX(self.n_sample-1, np.arange(self.data.n_node), self.data.G)
        ], 0)

        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.mu,
                [(np.repeat(self.data.Ts[None,self.data.G.N1(j)], self.n_sample, 0),
                  self.Xs_G[:,self.data.G.N2(j)], self.data.G.sub(self.data.G.N2(j)))
                 for j in np.arange(self.data.n_node)]
            ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j'))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.mu,
                    [(np.repeat(self.data.Ts[None,self.data.G.N1(j)], self.n_sample, 0),
                      self.Xs_G[:,self.data.G.N2(j)], self.data.G.sub(self.data.G.N2(j)))
                     for j in np.arange(self.data.n_node)]
                ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0)) 

        self.mus = np.array(r)
        
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.pi,
                [(np.repeat(self.data.Ts[None,self.data.G.N1(j)], self.n_sample, 0),
                  self.Xs_G[:,self.data.G.N2(j)], self.data.G.sub(self.data.G.N2(j)))
                 for j in np.arange(self.data.n_node)]
            ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j'))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.pi,
                    [(np.repeat(self.data.Ts[None,self.data.G.N1(j)], self.n_sample, 0),
                      self.Xs_G[:,self.data.G.N2(j)], self.data.G.sub(self.data.G.N2(j)))
                     for j in np.arange(self.data.n_node)]
                ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0)) 

        self.pis = np.array(r)
        
        self.ms = np.mean(self.mus, -1)
        self.varpis = np.mean(self.pis, -1)
        
    def mu(self, T_N1, X_N2, G_N2):
        return self.mu_fit.predict(T_N1, X_N2, G_N2)

    def pi(self, T_N1, X_N2, G_N2):
        return self.pi_fit.predict(T_N1, X_N2, G_N2)
    
    def rX(self, n_sample, N2, G):
        return self.cov_fit.sample(n_sample, N2, G)

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

    def dissimilarity(self, Y_j, T_N1j, Xs_N2j, G_N2j, T_N1k, Xs_N2k, G_N2k, hs=1, mode=0):
        hs = np.array(hs)
        h_shape = hs.shape
        hs = hs.flatten()
        
        Ts_N1j = np.repeat(T_N1j[None,:], Xs_N2j.shape[0], 0)
        Ts_N1k = np.repeat(T_N1k[None,:], Xs_N2k.shape[0], 0)
        Ds_N2j = self.delta(Ts_N1k, Xs_N2k, G_N2k, Ts_N1j, Xs_N2j, G_N2j)

        if self.nu_method == 'knn':
            hs = hs.astype(int)
            h_max = np.max(hs)
            
            proj_j = np.argpartition(Ds_N2j, hs, -1)[:,:h_max]
            D = np.cumsum(np.mean(
                Ds_N2j[np.repeat(np.arange(Xs_N2j.shape[0])[:,None], h_max, -1), proj_j], 0
            ))[hs-1]/hs

            if mode == 0:
                return D.reshape(h_shape)
        
            varpi_j = np.mean(self.pi(Ts_N1j, Xs_N2j, G_N2j))
            m_j = np.cumsum(np.mean(
                self.mu(Ts_N1j, Xs_N2j, G_N2j)[proj_j], 0
            ))[hs-1]/hs

            if mode == 1:
                return D.reshape(h_shape), m_j.reshape(h_shape)
        
            xi = (Y_j - self.mu(T_N1j, Xs_N2j[0], G_N2j)) \
               * varpi_j/self.pi(T_N1j, Xs_N2j[0], G_N2j) \
               * np.cumsum(np.sum(proj_j==0, 0))[hs-1]/hs \
               + m_j
            
        elif self.nu_method == 'ksm':
            Ws_N2j = np.exp(- hs[...,None,None] 
                            * Ds_N2j)
                            # * (Ds_N2j - np.min(Ds_N2j, -1)[...,None]))
            pnus_N2j = Ws_N2j / np.mean(Ws_N2j, -1)[...,None]
            nus_N2j = np.mean(pnus_N2j, -2)
            D = np.mean(Ds_N2j * pnus_N2j, (-2,-1))

            if mode == 0:
                return D.reshape(h_shape)
        
            varpi_j = np.mean(self.pi(Ts_N1j, Xs_N2j, G_N2j))
            m_j = np.mean(nus_N2j * self.mu(Ts_N1j, Xs_N2j, G_N2j), -1)

            if mode == 1:
                return D.reshape(h_shape), m_j.reshape(h_shape)
        
            xi = (Y_j - self.mu(T_N1j, Xs_N2j[0], G_N2j)) \
               * varpi_j/self.pi(T_N1j, Xs_N2j[0], G_N2j) \
               * nus_N2j[...,0] \
               + m_j
            
        else:
            raise('Only k-nearest-neighborhood (knn) and kernel smoothing (ksm) methods are supported now')
        
        return D.reshape(h_shape), xi.reshape(h_shape)

    def DR_estimate(self, i0, T0, Xs_N2i0=None, G0=None, 
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
                  self.Xs_G[:,self.data.G.N2(j)], self.data.G.sub(self.data.G.N2(j)),
                  T0_N1i0, Xs_N2i0, G0_N2i0, hs, mode)
                 for j in range(self.data.n_node))
            ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j'))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.dissimilarity, 
                    ((self.data.Ys[j], self.data.Ts[self.data.G.N1(j)],
                      self.Xs_G[:,self.data.G.N2(j)], self.data.G.sub(self.data.G.N2(j)),
                      T0_N1i0, Xs_N2i0, G0_N2i0, hs, mode) 
                     for j in range(self.data.n_node))
                ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0))

        Ds = np.array(r)[:,0,...]
        xis = np.array(r)[:,1,...]
        
        return np.sum(
            xis.reshape((self.data.n_node,)+(1,)*lamdas.ndim+hs.shape)
            * np.exp(- lamdas.reshape(lamdas.shape+(1,)*hs.ndim) 
                     * Ds.reshape((self.data.n_node,)+(1,)*lamdas.ndim+hs.shape)), 0
        ) / np.sum(
            np.exp(- lamdas.reshape(lamdas.shape+(1,)*hs.ndim) 
                   * Ds.reshape((self.data.n_node,)+(1,)*lamdas.ndim+hs.shape)), 0
        )

    def dissimilarity_new(self, j, Ts_N1k, Xs_N2k, G_N2k, hs=1, mode=0):
        hs = np.array(hs)
        h_shape = hs.shape
        hs = hs.flatten()

        Y_j = self.data.Ys[j]
        Ts_N1j = np.repeat(self.data.Ts[None, self.data.G.N1(j)], self.n_sample, 0)
        Xs_N2j = self.Xs_G[:,self.data.G.N2(j)]
        G_N2j = self.data.G.sub(self.data.G.N2(j))
        
        ds_j = self.delta(Ts_N1k, Xs_N2k, G_N2k, Ts_N1j, Xs_N2j, G_N2j)

        if self.nu_method == 'knn':
            hs = hs.astype(int)
            h_max = np.max(hs)
            
            proj_j = np.argpartition(ds_j, hs, -1)[:,:h_max]
            D = np.cumsum(np.mean(
                ds_j[np.repeat(np.arange(self.n_sample)[:,None], h_max, -1), proj_j], 0
            ))[hs-1]/hs

            if mode == 0:
                return D.reshape(h_shape)
        
            mn_j = np.cumsum(np.mean(
                self.mus[j,proj_j], 0
            ))[hs-1]/hs

            if mode == 1:
                return D.reshape(h_shape), mn_j.reshape(h_shape)
        
            xi = (Y_j - self.mus[j,0]) * self.varpis[j]/self.pis[j,0] \
               * np.cumsum(np.sum(proj_j==0, 0))[hs-1]/hs + mn_j
            
        elif self.nu_method == 'ksm':
            Ws_j = np.exp(- hs[...,None,None] 
                            # * ds_j)
                            * (ds_j - np.min(ds_j, -1)[...,None]))
            pnus_j = Ws_j / np.mean(Ws_j, -1)[...,None]
            nus_j = np.mean(pnus_j, -2)
            D = np.mean(ds_j * pnus_j, (-2,-1))

            if mode == 0:
                return D.reshape(h_shape)
        
            mn_j = np.mean(nus_j * self.mus[j], -1)

            if mode == 1:
                return D.reshape(h_shape), mn_j.reshape(h_shape)
        
            xi = (Y_j - self.mus[j,0]) * self.varpis[j]/self.pis[j,0] \
               * nus_j[...,0] + mn_j
            
        else:
            raise('Only k-nearest-neighborhood (knn) and kernel smoothing (ksm) methods are supported now')
        
        return D.reshape(h_shape), xi.reshape(h_shape)

    def DR_estimate_new(self, i0, T0, Xs_N2i0=None, G0=None, 
                        lamdas=1, hs=1, n_process=1, mode=2, 
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

        if Xs_N2i0 is None:
            Xs_N2i0 = self.rX(self.n_sample, N2i0, G0)

        T0s_N1i0 = np.repeat(T0[None,N1i0], Xs_N2i0.shape[0], 0)
        G0_N2i0 = G0.sub(N2i0)

        # def dissimilarity_new(self, j, Ts_N1k, Xs_N2k, G_N2k, hs=1, mode=0)
        
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.dissimilarity_new,
                ((j, T0s_N1i0, Xs_N2i0, G0_N2i0, hs, mode)
                 for j in range(self.data.n_node))
            ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j'))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.dissimilarity_new, 
                    ((j, T0s_N1i0, Xs_N2i0, G0_N2i0, hs, mode) 
                     for j in range(self.data.n_node))
                ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0))

        Ds = np.array(r)[:,0,...]
        xis = np.array(r)[:,1,...]
        
        return np.sum(
            xis.reshape((self.data.n_node,)+(1,)*lamdas.ndim+hs.shape)
            * np.exp(- lamdas.reshape(lamdas.shape+(1,)*hs.ndim) 
                     * Ds.reshape((self.data.n_node,)+(1,)*lamdas.ndim+hs.shape)), 0
        ) / np.sum(
            np.exp(- lamdas.reshape(lamdas.shape+(1,)*hs.ndim) 
                   * Ds.reshape((self.data.n_node,)+(1,)*lamdas.ndim+hs.shape)), 0
        )

###
### DR_quick requires too much memory 
### 
#     def DR_quick(self, i0, T0, Xs_N2i0=None, G0=None, 
#                  lamdas=1, hs=1, n_process=1, tqdm=None, level_tqdm=0):
#         if tqdm is None:
#             def tqdm(iterable, *args, **kwargs):
#                 return iterable
                
#         lamdas = np.array(lamdas)
#         hs = np.array(hs)

#         if G0 is None:
#             G0 = self.data.G

#         N1i0 = G0.N1(i0)
#         N2i0 = G0.N2(i0)
        
#         if Xs_N2i0 is None:
#             Xs_N2i0 = self.rX(self.n_sample, N2i0, G0)

#         T0s_N1i0 = np.repeat(T0[None,N1i0], Xs_N2i0.shape[0], 0)
#         G0_N2i0 = G0.sub(N2i0)
        
#         if n_process == 1:
#             from itertools import starmap
#             r = list(tqdm(starmap(self.delta,
#                 ((T0s_N1i0, Xs_N2i0, G0_N2i0, 
#                   np.repeat(self.data.Ts[None,self.data.G.N1(j)], self.n_sample, 0),
#                   self.Xs_G[:,self.data.G.N2(j)],
#                   self.data.G.sub(self.data.G.N2(j)))
#                  for j in range(self.data.n_node))
#             ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j'))
        
#         elif n_process > 1:
#             from multiprocessing import Pool
#             with Pool(n_process) as p:   
#                 r = list(tqdm(p.istarmap(self.delta,
#                     ((T0s_N1i0, Xs_N2i0, G0_N2i0, 
#                       np.repeat(self.data.Ts[None,self.data.G.N1(j)], self.n_sample, 0),
#                       self.Xs_G[:,self.data.G.N2(j)],
#                       self.data.G.sub(self.data.G.N2(j)))
#                      for j in range(self.data.n_node))
#                 ), total=self.data.n_node, leave=None, position=level_tqdm, desc='j', smoothing=0))
        
#         ds = np.array(r)
            
#         if self.nu_method == 'ksm':
#             Ws = np.exp(- hs[...,None,None,None] 
#                           * (ds - np.min(ds, -1)[...,None]))
#             pnus = Ws / np.mean(Ws, -1)[...,None]
#             nus = np.mean(pnus, -2)
            
#             mns = np.mean(nus * self.mus, -1)
#             xns = ((self.data.Ys - self.mus[:, 0]) * self.varpis
#                    / self.pis[:,0] * nus[...,0] + mns)
#             Ds = np.mean(ds * pnus, (-2, -1))

#         elif self.nu_method == 'knn':
#             hs = hs.astype(int)
#             h_max = np.max(hs)
            
#             proj = (np.argpartition(ds, hs, -1)[...,:h_max]).transpose((2,0,1))
            
#             mns = np.cumsum(np.mean(
#                 self.mus[np.arange(self.data.n_node)[:,None],proj], -1
#             ), 0)[hs-1]/hs[:,None]
#             xns = ((self.data.Ys - self.mus[:, 0]) * self.varpis / self.pis[:,0] 
#                    * (np.cumsum(np.sum(proj==0, -1), 0)[hs-1]/hs[:,None])
#                    + mns)
#             Ds = np.cumsum(np.mean(
#                 ds[np.arange(self.data.n_node)[:,None], 
#                    np.arange(self.n_sample), proj], -1
#             ), 0)[hs-1]/hs[:,None]

#         else:
#             raise('Only k-nearest-neighborhood (knn) and kernel smoothing (ksm) methods are supported now')
        
#         return np.sum(
#             xns
#             * np.exp(- lamdas.reshape(lamdas.shape+(1,)*(hs.ndim+1)) 
#                      * Ds), -1
#         ) / np.sum(
#             np.exp(- lamdas.reshape(lamdas.shape+(1,)*(hs.ndim+1)) 
#                    * Ds), -1
#         )

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
            ), total=self.data.n_node, leave=None, position=level_tqdm, desc='i0'))
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

        Xs_G = np.concatenate([
            self.data.Xs[None,...], self.rX(n_sample-1, np.arange(self.data.n_node), self.data.G)
        ], 0)
                
        ks_cv = random.choice(np.arange(self.data.n_node), n_cv, replace=False)
        
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.loo_cv_k,
                ((k, lamdas, hs, Xs_G[:,self.data.G.N2(k)],
                  n_sample, 1, tqdm, level_tqdm+1) 
                 for k in ks_cv)
            ), total=len(ks_cv), leave=None, position=level_tqdm, desc='k'))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.loo_cv_k,
                    ((k, lamdas, hs, Xs_G[:,self.data.G.N2(k)],
                      n_sample, 1, None, level_tqdm+1) 
                     for k in ks_cv)
                ), total=len(ks_cv), leave=None, position=level_tqdm, desc='k', smoothing=0))      

        return ks_cv, np.array([r_i[0] for r_i in r]), np.array([r_i[1] for r_i in r])

    def loo_cv_k(self, k, lamdas, hs, Xs_N2k, n_sample=100, n_process=1, 
                 tqdm = None, level_tqdm=0):
        N1k = self.data.G.N1(k)
        N2k = self.data.G.N2(k)
        
        Y_k = self.data.Ys[k]
        Ts_N1k = np.repeat(self.data.Ts[None,N1k], Xs_N2k.shape[0], 0)
        G_N2k = self.data.G.sub(N2k)
        
        varpi_k = np.mean(self.pi(Ts_N1k, Xs_N2k, G_N2k))
        m_k = np.mean(self.mu(Ts_N1k, Xs_N2k, G_N2k))

        xi = (Y_k - self.mu(Ts_N1k[0], Xs_N2k[0], G_N2k)) \
           * varpi_k/self.pi(Ts_N1k[0], Xs_N2k[0], G_N2k) \
           + m_k
        
        mk = np.delete(np.arange(self.data.n_node), N2k)
        
        Ys_mk = self.data.Ys[mk]
        Ts_mk = self.data.Ts[mk]
        Xs_mk = self.data.Xs[mk]
        G_mk = self.data.G.sub(mk)
        
        fit_mk = KECENI.Model(
            self.mu_model, self.pi_model, self.cov_model, self.delta, self.nu_method
        ).fit(KECENI.Data(Ys_mk, Ts_mk, Xs_mk, G_mk))

#         def DR_estimate(self, i0, T0, Xs_N2i0=None, G0=None, 
#                         lamdas=1, hs=1, n_sample=1000, n_process=1, mode=2, 
#                         tqdm=None, level_tqdm=0):

        xhat = fit_mk.DR_estimate(
            k, self.data.Ts, Xs_N2k, self.data.G, 
            lamdas=lamdas, hs=hs, n_sample=n_sample, n_process=n_process, mode=2,
            tqdm = tqdm, level_tqdm = level_tqdm
        )

        return xi, xhat

    def quick_cv(self, lamdas, hs, n_cv=100, n_process=1, 
               tqdm=None, level_tqdm=0):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
                        
        ks_cv = random.choice(np.arange(self.data.n_node), n_cv, replace=False)
        
        if n_process == 1:
            from itertools import starmap
            r = list(tqdm(starmap(self.quick_cv_k,
                ((k, lamdas, hs, tqdm, level_tqdm+1) 
                 for k in ks_cv)
            ), total=len(ks_cv), leave=None, position=level_tqdm, desc='k'))
        
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                r = list(tqdm(p.istarmap(self.quick_cv_k,
                    ((k, lamdas, hs, None, level_tqdm+1) 
                     for k in ks_cv)
                ), total=len(ks_cv), leave=None, position=level_tqdm, desc='k', smoothing=0))      

        return ks_cv, np.array([r_i[0] for r_i in r]), np.array([r_i[1] for r_i in r])

    def quick_cv_k(self, k, lamdas, hs, tqdm = None, level_tqdm=0):
        hs = np.array(hs)
        h_shape = hs.shape
        hs = hs.flatten()
        
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
                
        Ts_N1k = np.repeat(self.data.Ts[None,self.data.G.N1(k)], self.n_sample, 0)
        Xs_N2k = self.Xs_G[:,self.data.G.N2(k)]
        G_N2k = self.data.G.sub(self.data.G.N2(k))
        xi_k = ((self.data.Ys[k] - self.mus[k,0]) 
                * self.varpis[k] / self.pis[k,0] + self.ms[k])

        nodes_mk = np.delete(np.arange(self.data.n_node), self.data.G.N2(k))
        from itertools import starmap
        r = list(tqdm(starmap(self.delta,
            ((Ts_N1k, Xs_N2k, G_N2k, 
              np.repeat(self.data.Ts[None,self.data.G.N1(j)], self.n_sample, 0),
              self.Xs_G[:,self.data.G.N2(j)],
              self.data.G.sub(self.data.G.N2(j)))
             for j in nodes_mk)
        ), total=len(nodes_mk), leave=None, position=level_tqdm, desc='j'))
        ds = np.array(r)
            
        if self.nu_method == 'ksm':
            Ws = np.exp(- hs[...,None,None,None] 
                          * (ds - np.min(ds, -1)[...,None]))
            pnus = Ws / np.mean(Ws, -1)[...,None]
            nus = np.mean(pnus, -2)
            
            mns = np.mean(nus * self.mus[nodes_mk], -1)
            xns = ((self.data.Ys[nodes_mk] - self.mus[nodes_mk, 0]) 
                   * self.varpis[nodes_mk] / self.pis[nodes_mk,0] * nus[...,0] + mns)
            Ds = np.mean(ds * pnus, (-2, -1))

        elif self.nu_method == 'knn':
            hs = hs.astype(int)
            h_max = np.max(hs)
            
            proj = (np.argpartition(ds, hs, -1)[...,:h_max]).transpose((2,0,1))
            
            mns = np.cumsum(np.mean(
                self.mus[nodes_mk[:,None],proj], -1
            ), 0)[hs-1]/hs[:,None]
            xns = ((self.data.Ys[nodes_mk] - self.mus[nodes_mk, 0]) 
                   * self.varpis[nodes_mk] / self.pis[nodes_mk,0] 
                   * (np.cumsum(np.sum(proj==0, -1), 0)[hs-1]/hs[:,None])
                   + mns)
            Ds = np.cumsum(np.mean(
                ds[np.arange(len(nodes_mk))[:,None], 
                   np.arange(self.n_sample), proj], -1
            ), 0)[hs-1]/hs[:,None]

        else:
            raise('Only k-nearest-neighborhood (knn) and kernel smoothing (ksm) methods are supported now')

        xhat_k = np.sum(
            xns
            * np.exp(- lamdas.reshape(lamdas.shape+(1,)*(hs.ndim+1)) 
                     * Ds), -1
        ) / np.sum(
            np.exp(- lamdas.reshape(lamdas.shape+(1,)*(hs.ndim+1)) 
                   * Ds), -1
        )

        return xi_k, xhat_k.reshape(lamdas.shape+h_shape)