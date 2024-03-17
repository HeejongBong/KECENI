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

    def sample(self, n_sample, X_N2, G_N2):
        n1 = len(G_N2.N1(0))
        return np.zeros((n_sample, n1))



###
class FittedPropensityModel(PropensityModel):
    def __init__(self, pi, rT=None):
        self.pi = pi
        self.rT = rT

    def fit(self, data):
        return(FittedPropensityFit(self.pi, self.rT))

class FittedPropensityFit(PropensityFit):
    def __init__(self, pi, rT):
        self.pi = pi
        self.rT = rT

    def predict(self, T_N1, X_N2, G_N2):
        return self.pi(T_N1, X_N2, G_N2)

    def sample(self, n_sample, X_N2, G_N2):
        return self.rT(n_sample, X_N2, G_N2)




###
class IIDPropensityModel(PropensityModel):
    def fit(self, data):
        return IIDPropensityFit()
        
class IIDPropensityFit(PropensityFit):
    def predict_i(self, T, X_N1, G_N1):
        return 0

    def predict(self, T_N1, X_N2, G_N2):
        n1 = len(G_N2.N1(0)); n2 = len(G_N2.N2(0))
        return np.prod([self.predict_i(
            T_N1[...,j], X_N2[...,G_N2.N1(j),:], G_N2.sub(G_N2.N1(j))
        ) for j in G_N2.N1(0)], 0)

    def sample_i(self, n_sample, X_N1, G_N1):
        return np.zeros(n_sample)

    def sample(self, n_sample, X_N2, G_N2):
        n1 = len(G_N2.N1(0)); n2 = len(G_N2.N2(0))
        return np.stack([self.sample_i(
            n_sample,  X_N2[...,G_N2.N1(j),:], G_N2.sub(G_N2.N1(j))
        ) for j in G_N2.N1(0)], -1)





###
class LinearIIDPropensityModel(IIDPropensityModel):
    def __init__(self, summary, *args, **kwargs):
        self.summary = summary
        self.model = LinearRegression(*args, **kwargs)

    def fit(self, data):
        Zs = np.array([
            self.summary(data.Xs[data.G.N1(j)],
                         data.G.sub(data.G.N1(j)))
            for j in np.arange(data.n_node)])

        model_fit = self.model.fit(Zs, data.Ts)
        model_fit.sigma_ = np.sqrt(np.mean(
            (data.Ts - model_fit.predict(Zs))**2
        ))
        return LinearIIDPropensityFit(self.summary, model_fit)
        
class LinearIIDPropensityFit(IIDPropensityFit):
    def __init__(self, summary, model_fit):
        self.summary = summary
        self.model_fit = model_fit
    
    def predict_i(self, T, X_N1, G_N1):
        Z = self.summary(X_N1, G_N1)
        dZ = Z.shape[-1]
        T_hat = self.model_fit.predict(Z.reshape([-1,dZ])).reshape(Z.shape[:-1])
        return stats.norm.pdf((T - T_hat)/self.model_fit.sigma_)

    def sample_i(self, n_sample, X_N1, G_N1):
        Z = self.summary(X_N1, G_N1)
        dZ = Z.shape[-1]
        T_hat = self.model_fit.predict(Z.reshape([-1,dZ])).reshape(Z.shape[:-1])
        return stats.norm.rvs(T_hat, self.model_fit.sigma_, size=(n_sample,)+T_hat.shape)




###
class LogisticIIDPropensityModel(IIDPropensityModel):
    def __init__(self, summary, *args, **kwargs):
        self.summary = summary
        self.model = LogisticRegression(*args, **kwargs)

    def fit(self, data):
        Zs = np.array([
            self.summary(data.Xs[data.G.N1(j)],
                         data.G.sub(data.G.N1(j)))
            for j in np.arange(data.n_node)])
        
        model_fit = self.model.fit(Zs, data.Ts)
        return LogisticIIDPropensityFit(self.summary, model_fit)

class LogisticIIDPropensityFit(IIDPropensityFit):
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

    def sample_i(self, n_sample, X_N1, G_N1):
        Z = self.summary(X_N1, G_N1)
        dZ = Z.shape[-1]
        return random.binomial(1, self.model_fit.predict_proba(Z.reshape([-1,dZ]))[:,0].reshape(Z.shape[:-1]),
                               size=(n_sample,)+Z.shape[:-1])




###
class KernelIIDPropensityModel(IIDPropensityModel):
    def __init__(self, delta, lamda=None, *args, **kwargs):
        self.delta = delta
        self.lamda = lamda

    def fit(self, data):
        return KernelIIDPropensityFit(self.delta, self.lamda, data)

class KernelIIDPropensityFit(IIDPropensityFit):
    def __init__(self, delta, lamda, data):
        self.delta = delta
        self.lamda = lamda
        self.data = data

    def loo_cv(self, lamdas, n_cv=100, n_sample=100, n_process=1, 
               tqdm=None, leave_tqdm=True):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable
                
        ks_cv = random.choice(np.arange(self.data.n_node), n_cv, replace=False)
        
        if n_process == 1:
            from itertools import starmap
            Ys_cv = list(tqdm(starmap(self.loo_cv_k,
                ((k, lamdas, n_sample, 1, None, False) 
                 for k in ks_cv)
            ), total=len(ks_cv), leave=leave_tqdm, desc='j'))
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                Ys_cv = list(tqdm(p.istarmap(self.loo_cv_k,
                    ((k, lamdas, n_sample, 1, None, False) 
                     for k in ks_cv)
                ), total=len(ks_cv), leave=leave_tqdm, desc='j'))

        return ks_cv, np.array(Ys_cv).T

    def loo_cv_k(self, k, lamdas, n_sample=100, n_process=1, 
                 tqdm = None, leave_tqdm=False):
        N1k = self.data.G.N1(k)
        N2k = self.data.G.N2(k)
        mk = np.delete(np.arange(self.data.n_node), N2k)
        
        data_mk = KECENI.Data(
            self.data.Ys[mk], self.data.Ts[mk], 
            self.data.Xs[mk], self.data.G.sub(mk)
        )
        fit_mk = KernelIIDPropensityModel(
            self.delta
        ).fit(data_mk)
    
        return fit_mk.predict_i(
            self.data.Ts[k], self.data.Xs[N1k], 
            self.data.G.sub(N1k), lamdas=lamdas
        )

    def predict_i(self, T, X_N1, G_N1, lamdas):
        if lamdas is None:
            lamdas = np.array(self.lamda)
        else:
            lamdas = np.array(lamdas)
        Ds = np.array(
            [self.delta(X_N1, G_N1, 
                        self.data.Xs[self.data.G.N1(i)],
                        self.data.G.sub(self.data.G.N1(i)))
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

    def sample_i(self, n_sample, X_N1, G_N1):
        ##############
        #### TODO ####
        ##############
        return np.zeros(n_sample)