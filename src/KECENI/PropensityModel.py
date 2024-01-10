import numpy as np
import numpy.random as random
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, LogisticRegression

import KECENI



###
class PropensityModel:
    def __init__(self):
        pass
        
    def fit(self, data):
        return PropensityFit()
        
class PropensityFit:
    def __init__(self):
        pass

    def predict(self, T_N1, X_N2, G_N2):
        return 0



###
class FittedPropensityModel(PropensityModel):
    def __init__(self, pi):
        self.pi = pi

    def fit(self, data):
        return(FittedPropensityFit(self.pi))

class FittedPropensityFit(PropensityFit):
    def __init__(self, pi):
        self.pi = pi

    def predict(self, T_N1, X_N2, G_N2):
        return self.pi(T_N1, X_N2, G_N2)



###
class LinearIIDPropensityModel(PropensityModel):
    def __init__(self, summary, *args, **kwargs):
        self.summary = summary
        self.model = LinearRegression(*args, **kwargs)

    def fit(self, data):
        Zs = np.array([
            self.summary(data.Xs[data.N1s[j]],
                         data.G[np.ix_(data.N1s[j],data.N1s[j])])
            for j in np.arange(data.n_node)])

        model_fit = self.model.fit(Zs, data.Ts)
        model_fit.sigma_ = np.sqrt(np.mean(
            (data.Ts - model_fit.predict(Zs))**2
        ))
        return LinearIIDPropensityFit(self.summary, model_fit)
        
class LinearIIDPropensityFit(PropensityFit):
    def __init__(self, summary, model_fit):
        self.summary = summary
        self.model_fit = model_fit
    
    def predict_i(self, T, X_N1, G_N1):
        Z = self.summary(X_N1, G_N1)
        dZ = Z.shape[-1]
        T_hat = self.model_fit.predict(Z.reshape([-1,dZ])).reshape(Z.shape[:-1])
        return stats.norm.pdf((T - T_hat)/self.model_fit.sigma_)
        
    def predict(self, T_N1, X_N2, G_N2):
        n1 = T_N1.shape[-1]; n2 = X_N2.shape[-2]
        N1s = [np.concatenate([[j],np.nonzero(G_N2[j])[0]]) 
               for j in np.arange(n2)]
        return np.prod([self.predict_i(
            T_N1[...,j], X_N2[...,N1s[j],:], G_N2[np.ix_(N1s[j],N1s[j])]
        ) for j in np.arange(n1)], 0)




###
class LogisticIIDPropensityModel(PropensityModel):
    def __init__(self, summary, *args, **kwargs):
        self.summary = summary
        self.model = LogisticRegression(*args, **kwargs)

    def fit(self, data):
        Zs = np.array([
            self.summary(data.Xs[data.N1s[j]],
                         data.G[np.ix_(data.N1s[j],data.N1s[j])])
            for j in np.arange(data.n_node)])
        
        model_fit = self.model.fit(Zs, data.Ts)
        return LogisticIIDPropensityFit(self.summary, model_fit)

class LogisticIIDPropensityFit(PropensityFit):
    def __init__(self, summary, model_fit):
        self.summary = summary
        self.model_fit = model_fit

    def predict_i(self, T, X_N1, G_N1):
        Z = self.summary(X_N1, G_N1)
        dZ = Z.shape[-1]
        return np.abs(
            self.model_fit.predict_proba(Z.reshape([-1,dZ]))[:,0]
            - T.flatten()
        ).reshape(T.shape)
    
    def predict(self, T_N1, X_N2, G_N2):
        n1 = T_N1.shape[-1]; n2 = X_N2.shape[-2]
        N1s = [np.concatenate([[j],np.nonzero(G_N2[j])[0]]) 
               for j in np.arange(n2)]
        return np.prod([self.predict_i(
            T_N1[...,j], X_N2[...,N1s[j],:], G_N2[np.ix_(N1s[j],N1s[j])]
        ) for j in np.arange(n1)], 0)




###
class KernelIIDPropensityModel(PropensityModel):
    def __init__(self, delta, *args, **kwargs):
        self.delta = delta

    def fit(self, data):
        return KernelIIDPropensityFit(self.delta, data)

class KernelIIDPropensityFit(PropensityFit):
    def __init__(self, delta, data):
        self.delta = delta
        self.data = data

    def loo_cv_old(self, lamdas, n_cv=100, tqdm=None, leave_tqdm=False):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
                
        model = KernelRegressionModel(self.delta)

        ks_cv = random.choice(np.arange(self.data.n_node), n_cv)
        Ys_cv = np.zeros(lamdas.shape + (n_cv,))

        for iter_k in tqdm(range(n_cv), leave=leave_tqdm, desc='k', total=n_cv):
            k = ks_cv[iter_k]
            N1k = self.data.N1s[k]
            N2k = self.data.N2s[k]
            
            mk = np.delete(np.arange(self.data.n_node), self.data.N2s[k])
            
            data_mk = KECENI.Data(
                self.data.Ys[mk], self.data.Ts[mk], 
                self.data.Xs[mk], self.data.G[np.ix_(mk,mk)]
            )
            fit_mk = model.fit(data_mk)
        
            Ys_cv[:,iter_k] = fit_mk.predict(
                self.data.Ts[k], self.data.Xs[N1k], 
                self.data.G[np.ix_(N1k,N1k)], lamdas=lamdas
            )

        return ks_cv, Ys_cv

    def loo_cv(self, lamdas, n_cv=100, n_sample=100, n_process=1, 
               tqdm=None, leave_tqdm=True):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
                
        ks_cv = random.choice(np.arange(self.data.n_node), n_cv, replace=False)
        
        if n_process == 1:
            from itertools import starmap
            Ys_cv = list(starmap(self.loo_cv_k, tqdm(
                ((k, lamdas, n_sample, 1, None, False) 
                 for k in ks_cv),
                total=len(ks_cv), leave=leave_tqdm, desc='j'
            )))
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                Ys_cv = list(p.starmap(self.loo_cv_k, tqdm(
                    ((k, lamdas, n_sample, 1, None, False) 
                     for k in ks_cv),
                    total=len(ks_cv), leave=leave_tqdm, desc='j'
                )))

        return ks_cv, np.array(Ys_cv).T

    def loo_cv_k(self, k, lamdas, n_sample=100, n_process=1, 
                 tqdm = None, leave_tqdm=False):
        N1k = self.data.N1s[k]
        N2k = self.data.N2s[k]
        mk = np.delete(np.arange(self.data.n_node), N2k)
        
        data_mk = KECENI.Data(
            self.data.Ys[mk], self.data.Ts[mk], 
            self.data.Xs[mk], self.data.G[np.ix_(mk,mk)]
        )
        fit_mk = KernelIIDPropensityModel(
            self.delta
        ).fit(data_mk)
    
        return fit_mk.predict_i(
            self.data.Ts[k], self.data.Xs[N1k], 
            self.data.G[np.ix_(N1k,N1k)], lamdas=lamdas
        )

    def predict_i(self, T, X_N1, G_N1, lamdas):
        if lamdas is None:
            lamdas = np.array(self.lamda)
        else:
            lamdas = np.array(lamdas)
        Ds = np.array(
            [self.delta(X_N1, G_N1, 
                        self.data.Xs[self.data.N1s[i]],
                        self.data.G[np.ix_(self.data.N1s[i],self.data.N1s[i])])
             for i in np.arange(self.data.n_node)]
        ).T
        return np.abs(
            np.sum(self.data.Ts
                   * np.exp(- lamdas.reshape(lamdas.shape+(1,)*Ds.ndim) 
                            * Ds), -1) \
            / np.sum(np.exp(- lamdas.reshape(lamdas.shape+(1,)*Ds.ndim) 
                           * Ds), -1)
            - 1 + T
        )
    
    def predict(self, T_N1, X_N2, G_N2, lamdas=None):        
        n1 = T_N1.shape[-1]; n2 = X_N2.shape[-2]
        N1s = [np.concatenate([[j],np.nonzero(G_N2[j])[0]]) 
               for j in np.arange(n2)]
        return np.prod([self.predict_i(
            T_N1[...,j], X_N2[...,N1s[j],:], G_N2[np.ix_(N1s[j],N1s[j])],
            lamdas
        ) for j in np.arange(n1)], 0)