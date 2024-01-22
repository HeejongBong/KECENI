import numpy as np
import numpy.random as random
from sklearn.linear_model import LinearRegression, LogisticRegression

import KECENI

###
class RegressionModel:
    def __init__(self):
        pass
        
    def fit(self, data):
        return RegressionFit()

class RegressionFit:
    def __init__(self):
        pass

    def predict(self, T_N1, X_N2, G_N2):
        pass





###
class FittedRegressionModel(RegressionModel):
    def __init__(self, mu):
        self.mu = mu

    def fit(self, data):
        return FittedRegressionFit(self.mu)
        
class FittedRegressionFit(RegressionFit):
    def __init__(self, mu):
        self.mu = mu

    def predict(self, T_N1, X_N2, G_N2):
        return self.mu(T_N1, X_N2, G_N2)    
        



###
class LinearRegressionModel(RegressionModel):
    def __init__(self, summary, *args, **kwargs):
        self.summary = summary
        self.model = LinearRegression(*args, **kwargs)

    def fit(self, data):
        Zs = np.array([
            self.summary(data.Ts[data.G.N1(j)],
                         data.Xs[data.G.N2(j)],
                         data.G.sub(data.G.N2(j)))
            for j in np.arange(data.n_node)])

        model_fit = self.model.fit(Zs, data.Ys)
        model_fit.sigma_ = np.sqrt(np.mean((data.Ys - model_fit.predict(Zs))**2))
        return LinearRegressionFit(self.summary, model_fit)

class LinearRegressionFit(RegressionFit):
    def __init__(self, summary, model_fit):
        self.summary = summary
        self.model_fit = model_fit
        
    def predict(self, T_N1, X_N2, G_N2):
        Z = self.summary(T_N1, X_N2, G_N2)
        dZ = Z.shape[-1]
        return self.model_fit.predict(Z.reshape([-1,dZ])).reshape(Z.shape[:-1])



###
class LogisticRegressionModel(RegressionModel):
    def __init__(self, summary, *args, **kwargs):
        self.summary = summary
        self.model = LogisticRegression(*args, **kwargs)

    def fit(self, data):
        Zs = np.array([
            self.summary(data.Ts[data.G.N1(j)],
                         data.Xs[data.G.N2(j)],
                         data.G.sub(data.G.N2(j)))
            for j in np.arange(data.n_node)])
        
        model_fit = self.model.fit(Zs, data.Ys)
        return LogisticRegressionFit(self.summary, model_fit)

class LogisticRegressionFit(RegressionFit):
    def __init__(self, summary, model_fit):
        self.summary = summary
        self.model_fit = model_fit

    def predict(self, T_N1, X_N2, G_N2):
        Z = self.summary(T_N1, X_N2, G_N2)
        dZ = Z.shape[-1]
        return self.model_fit.predict_proba(Z.reshape([-1,dZ]))[:,-1].reshape(Z.shape[:-1])




###
class KernelRegressionModel(RegressionModel):
    def __init__(self, delta, *args, **kwargs):
        self.delta = delta

    def fit(self, data):
        return KernelRegressionFit(self.delta, data)

class KernelRegressionFit(RegressionFit):
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
            N1k = self.data.G.N1(k)
            N2k = self.data.G.N2(k)
            
            mk = np.delete(np.arange(self.data.n_node), N2k)
            
            data_mk = KECENI.Data(
                self.data.Ys[mk], self.data.Ts[mk], 
                self.data.Xs[mk], self.data.G.sub(mk)
            )
            fit_mk = model.fit(data_mk)
        
            Ys_cv[:,iter_k] = fit_mk.predict(
                self.data.Ts[N1k], self.data.Xs[N2k], 
                self.data.G.sub(N2k), lamdas=lamdas
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
        fit_mk = KernelRegressionModel(
            self.delta
        ).fit(data_mk)
    
        return fit_mk.predict(
            self.data.Ts[N1k], self.data.Xs[N2k], 
            self.data.G.sub(N2k), lamdas=lamdas
        )

    def predict(self, T_N1, X_N2, G_N2, lamdas=None):
        if lamdas is None:
            lamdas = np.array(self.lamda)
        else:
            lamdas = np.array(lamdas)
        Ds = np.array(
            [self.delta(T_N1, X_N2, G_N2, 
                        self.data.Ts[self.data.G.N1(i)],
                        self.data.Xs[self.data.G.N2(i)],
                        self.data.G.sub(self.data.G.N2(i)))
             for i in np.arange(self.data.n_node)]
        ).T

        return np.sum(self.data.Ys
                      * np.exp(- lamdas.reshape(lamdas.shape+(1,)*Ds.ndim) 
                               * Ds), -1) \
               / np.sum(np.exp(- lamdas.reshape(lamdas.shape+(1,)*Ds.ndim) 
                               * Ds), -1)