import numpy as np
import numpy.random as random
import scipy.linalg as la
from sklearn.linear_model import LinearRegression, LogisticRegression

from .Data import Data

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
    
    def H(self, T_N1, X_N2, G_N2):
        pass
    
    def predict_with_H(self, T_N1, X_N2, G_N2):
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
        mu = self.mu(T_N1, X_N2, G_N2) 
        return mu
    
    def H(self, T_N1, X_N2, G_N2):
        return np.zeros(np.braodcast_shapes(T_N1.shape[:-1],X_N2.shape[:-2])+(1,))
    
    def predict_with_H(self, T_N1, X_N2, G_N2):
        mu = self.mu(T_N1, X_N2, G_N2) 
        return mu, np.zeros(mu.shape+(1,))
        



###
class LinearRegressionModel(RegressionModel):
    def __init__(self, summary, *args, **kwargs):
        self.summary = summary
        self.model = LinearRegression(fit_intercept=False, *args, **kwargs)

    def fit(self, data):
        Zs = np.array([
            self.summary(data.Ts[data.G.N1(j)],
                         data.Xs[data.G.N2(j)],
                         data.G.sub(data.G.N2(j)))
            for j in np.arange(data.n_node)])
        Zs = np.concatenate([np.full(Zs.shape[:-1]+(1,), 1), Zs], -1)

        model_fit = self.model.fit(Zs, data.Ys)
        model_fit.Zs_ = Zs
        model_fit.Ys_ = data.Ys
        model_fit.residuals_ = data.Ys - model_fit.predict(Zs)
        model_fit.sigma_ = np.sqrt(np.mean((model_fit.residuals_)**2))
        return LinearRegressionFit(self.summary, model_fit)

class LinearRegressionFit(RegressionFit):
    def __init__(self, summary, model_fit):
        self.summary = summary
        self.model_fit = model_fit
        
    def predict(self, T_N1, X_N2, G_N2):
        Z = self.summary(T_N1, X_N2, G_N2)
        Z = np.concatenate([np.full(Z.shape[:-1]+(1,), 1), Z], -1)
        dZ = Z.shape[-1]
        
        return self.model_fit.predict(Z.reshape([-1,dZ])).reshape(Z.shape[:-1])
    
    def H(self, T_N1, X_N2, G_N2):
        Z = self.summary(T_N1, X_N2, G_N2)
        Z = np.concatenate([np.full(Z.shape[:-1]+(1,), 1), Z], -1)
        dZ = Z.shape[-1]

        return (
            Z @ la.inv(self.model_fit.Zs_.T @ self.model_fit.Zs_)
            @ (self.model_fit.Zs_.T * self.model_fit.residuals_)
        )

    def predict_with_H(self, T_N1, X_N2, G_N2):
        Z = self.summary(T_N1, X_N2, G_N2)
        Z = np.concatenate([np.full(Z.shape[:-1]+(1,), 1), Z], -1)
        dZ = Z.shape[-1]

        mu = self.model_fit.predict(Z.reshape([-1,dZ])).reshape(Z.shape[:-1])
        H = (
            Z @ la.inv(self.model_fit.Zs_.T @ self.model_fit.Zs_)
            @ (self.model_fit.Zs_.T * self.model_fit.residuals_)
        )

        return mu, H



###
class LogisticRegressionModel(RegressionModel):
    def __init__(self, summary, *args, **kwargs):
        self.summary = summary
        self.model = LogisticRegression(penalty=None, fit_intercept=False,
                                        *args, **kwargs)

    def fit(self, data):
        Zs = np.array([
            self.summary(data.Ts[data.G.N1(j)],
                         data.Xs[data.G.N2(j)],
                         data.G.sub(data.G.N2(j)))
            for j in np.arange(data.n_node)])
        Zs = np.concatenate([np.full(Zs.shape[:-1]+(1,), 1), Zs], -1)
        
        model_fit = self.model.fit(Zs, data.Ys)
        model_fit.Zs_ = Zs
        model_fit.Ys_ = data.Ys
        model_fit.residuals_ = data.Ys - model_fit.predict_proba(Zs)[:,1]
        model_fit.var_ = (
            np.abs(model_fit.residuals_) * (1 - np.abs(model_fit.residuals_))
        )
        return LogisticRegressionFit(self.summary, model_fit)

class LogisticRegressionFit(RegressionFit):
    def __init__(self, summary, model_fit):
        self.summary = summary
        self.model_fit = model_fit

    def predict(self, T_N1, X_N2, G_N2):
        Z = self.summary(T_N1, X_N2, G_N2)
        Z = np.concatenate([np.full(Z.shape[:-1]+(1,), 1), Z], -1)
        dZ = Z.shape[-1]
        return self.model_fit.predict_proba(Z.reshape([-1,dZ]))[:,-1].reshape(Z.shape[:-1])
    
    def H(self, T_N1, X_N2, G_N2):
        Z = self.summary(T_N1, X_N2, G_N2)
        Z = np.concatenate([np.full(Z.shape[:-1]+(1,), 1), Z], -1)
        dZ = Z.shape[-1]

        mu = self.model_fit.predict_proba(Z.reshape([-1,dZ]))[:,-1].reshape(Z.shape[:-1])
        var = (1 - mu) * mu
        return (
            (var[...,None] * Z) 
            @ la.pinv((self.model_fit.Zs_.T * self.model_fit.var_)
                      @ self.model_fit.Zs_)
            @ (self.model_fit.Zs_.T * self.model_fit.residuals_)
        )
        
    def predict_with_H(self, T_N1, X_N2, G_N2):
        Z = self.summary(T_N1, X_N2, G_N2)
        Z = np.concatenate([np.full(Z.shape[:-1]+(1,), 1), Z], -1)
        dZ = Z.shape[-1]

        mu = self.model_fit.predict_proba(Z.reshape([-1,dZ]))[:,-1].reshape(Z.shape[:-1])
        var = (1 - mu) * mu
        H = (
            (var[...,None] * Z) 
            @ la.pinv((self.model_fit.Zs_.T * self.model_fit.var_)
                      @ self.model_fit.Zs_)
            @ (self.model_fit.Zs_.T * self.model_fit.residuals_)
        )
        
        return mu, H




###
class KernelRegressionModel(RegressionModel):
    def __init__(self, delta, lamda=None, ths=1e-3, *args, **kwargs):
        self.delta = delta
        self.lamda = lamda
        self.clip = - np.log(1e-3)

    def fit(self, data):
        return KernelRegressionFit(self.delta, self.lamda, self.clip, data)

class KernelRegressionFit(RegressionFit):
    def __init__(self, delta, lamda, clip, data):
        self.delta = delta
        self.lamda = lamda
        self.data = data
        self.clip = clip


    def loo_cv(self, lamdas, i0s=None, n_process=1, tqdm=None, leave_tqdm=True):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable

        if i0s is None:
            i0s = np.arange(self.data.n_node)
        
        # ks_cv = random.choice(np.arange(self.data.n_node), n_cv, replace=False)
        
        if n_process == 1:
            from itertools import starmap
            mus_cv = list(tqdm(starmap(self.loo_cv_k,
                ((k, lamdas, 1, None, False) 
                 for k in i0s)     
            ), total=len(i0s), leave=leave_tqdm, desc='j', smoothing=0))
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                mus_cv = list(tqdm(p.istarmap(self.loo_cv_k,
                    ((k, lamdas, 1, None, False) 
                     for k in i0s)
                ), total=len(i0s), leave=leave_tqdm, desc='j', smoothing=0))

        return self.data.Ys[i0s], np.array(mus_cv).T

    def loo_cv_k(self, k, lamdas, n_process=1, tqdm = None, leave_tqdm=False):
        N1k = self.data.G.N1(k)
        N2k = self.data.G.N2(k)
        mk = np.delete(np.arange(self.data.n_node), self.data.G.N2(k))
        
        # data_mk = Data(
        #     self.data.Ys[mk], self.data.Ts[mk], 
        #     self.data.Xs[mk], self.data.G.sub(mk)
        # )
        # fit_mk = KernelRegressionModel(
        #     self.delta
        # ).fit(data_mk)
    
        # return fit_mk.predict(
        #     self.data.Ts[N1k], self.data.Xs[N2k], 
        #     self.data.G.sub(N2k), lamdas=lamdas
        # )

        Ds = np.stack(
            [self.delta(self.data.Ts[N1k], 
                        self.data.Xs[N2k], 
                        self.data.G.sub(N2k),
                        self.data.Ts[self.data.G.N1(i)],
                        self.data.Xs[self.data.G.N2(i)],
                        self.data.G.sub(self.data.G.N2(i)))
             for i in mk], -1
        )
        Ds = Ds - np.min(Ds, -1)[...,None]

        lamDs = lamdas.reshape(lamdas.shape+(1,)*Ds.ndim) * Ds
        ws = np.zeros(lamDs.shape)
        ws[lamDs < self.clip] = np.exp(- lamDs[lamDs < self.clip])

        return np.sum(self.data.Ys[mk] * ws, -1) / np.sum(ws, -1)

    def predict(self, T_N1, X_N2, G_N2, lamdas=None):
        if lamdas is None:
            lamdas = np.array(self.lamda)
        else:
            lamdas = np.array(lamdas)
            
        Ds = np.stack(
            [self.delta(T_N1, X_N2, G_N2, 
                        self.data.Ts[self.data.G.N1(i)],
                        self.data.Xs[self.data.G.N2(i)],
                        self.data.G.sub(self.data.G.N2(i)))
             for i in np.arange(self.data.n_node)], -1
        )
        Ds = Ds - np.min(Ds, -1)[...,None]

        lamDs = lamdas.reshape(lamdas.shape+(1,)*Ds.ndim) * Ds
        ws = np.zeros(lamDs.shape)
        ws[lamDs < self.clip] = np.exp(- lamDs[lamDs < self.clip])

        mus = np.sum(self.data.Ys * ws, -1) / np.sum(ws, -1)

        return mus
    
    def H(self, T_N1, X_N2, G_N2, lamdas=None):
        if lamdas is None:
            lamdas = np.array(self.lamda)
        else:
            lamdas = np.array(lamdas)
            
        Ds = np.stack(
            [self.delta(T_N1, X_N2, G_N2, 
                        self.data.Ts[self.data.G.N1(i)],
                        self.data.Xs[self.data.G.N2(i)],
                        self.data.G.sub(self.data.G.N2(i)))
             for i in np.arange(self.data.n_node)], -1
        )
        Ds = Ds - np.min(Ds, -1)[...,None]

        lamDs = lamdas.reshape(lamdas.shape+(1,)*Ds.ndim) * Ds
        ws = np.zeros(lamDs.shape)
        ws[lamDs < self.clip] = np.exp(- lamDs[lamDs < self.clip])

        mu = np.sum(self.data.Ys * ws, -1) / np.sum(ws, -1)
        return (
            (self.data.Ys - mu[...,None]) * ws 
            / np.sum(ws, -1)[...,None]
        )

    def predict_with_H(self, T_N1, X_N2, G_N2, lamdas=None):
        if lamdas is None:
            lamdas = np.array(self.lamda)
        else:
            lamdas = np.array(lamdas)
            
        Ds = np.stack(
            [self.delta(T_N1, X_N2, G_N2, 
                        self.data.Ts[self.data.G.N1(i)],
                        self.data.Xs[self.data.G.N2(i)],
                        self.data.G.sub(self.data.G.N2(i)))
             for i in np.arange(self.data.n_node)], -1
        )
        Ds = Ds - np.min(Ds, -1)[...,None]

        lamDs = lamdas.reshape(lamdas.shape+(1,)*Ds.ndim) * Ds
        ws = np.zeros(lamDs.shape)
        ws[lamDs < self.clip] = np.exp(- lamDs[lamDs < self.clip])

        mu = np.sum(self.data.Ys * ws, -1) / np.sum(ws, -1)
        H = (
            (self.data.Ys - mu[...,None]) * ws 
            / np.sum(ws, -1)[...,None]
        )

        return mu, H