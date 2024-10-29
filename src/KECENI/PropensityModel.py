import numpy as np
import numpy.random as random
import scipy.stats as stats
import scipy.linalg as la
from sklearn.linear_model import LinearRegression, LogisticRegression

from .Data import Data



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
        pass

    def sample(self, n_sample, X_N2, G_N2):
        pass



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
    
    def predict_with_residual(self, T_N1, X_N2, G_N2):
        pi = self.pi(T_N1, X_N2, G_N2)
        return pi, np.zeros(pi.shape+(1,))

    def sample(self, n_sample, X_N2, G_N2):
        return self.rT(n_sample, X_N2, G_N2)




###
class IIDPropensityModel(PropensityModel):
    def fit(self, data):
        return IIDPropensityFit()
        
class IIDPropensityFit(PropensityFit):
    def predict_i(self, T, X_N1, G_N1):
        T = np.array(T)
        pi_i = np.full(
            np.broadcast_shapes(T.shape, X_N1.shape[:-2]),
            0.5
        )
        return pi_i

    def predict_with_residual_i(self, T, X_N1, G_N1):
        T = np.array(T)
        pi_i = np.full(
            np.broadcast_shapes(T.shape, X_N1.shape[:-2]),
            0.5
        )
        res_i = np.zeros(pi_i.shape+(1,))
        return pi_i, res_i

    def predict(self, T_N1, X_N2, G_N2, *args, **kwargs):
        n1 = len(G_N2.N1(0)); n2 = len(G_N2.N2(0))
        return np.prod([self.predict_i(
            T_N1[...,j], X_N2[...,G_N2.N1(j),:], G_N2.sub(G_N2.N1(j)),
            *args, **kwargs
        ) for j in G_N2.N1(0)], 0)

    def predict_with_residual(self, T_N1, X_N2, G_N2, *args, **kwargs):
        n1 = len(G_N2.N1(0)); n2 = len(G_N2.N2(0))

        pi_is, res_is = list(zip(*[self.predict_with_residual_i(
            T_N1[...,j], X_N2[...,G_N2.N1(j),:], G_N2.sub(G_N2.N1(j)),
            *args, **kwargs
        ) for j in G_N2.N1(0)]))
        pi_is = np.array(pi_is); res_is = np.array(res_is)

        pi = np.prod(pi_is, 0)
        res = np.sum((pi / pi_is)[...,None] * res_is, 0)
        
        return pi, res

    def sample_i(self, n_sample, X_N1, G_N1, *args, **kwargs):
        ps = self.predict_i(1, X_N1, G_N1, *args, **kwargs)
        return random.binomial(1, ps, size=(n_sample,)+ps.shape)

    def sample(self, n_sample, X_N2, G_N2, *args, **kwargs):
        n1 = len(G_N2.N1(0)); n2 = len(G_N2.N2(0))
        return np.stack([self.sample_i(
            n_sample,  X_N2[...,G_N2.N1(j),:], G_N2.sub(G_N2.N1(j)),
            *args, **kwargs
        ) for j in G_N2.N1(0)], -1)





# ###
# class LinearIIDPropensityModel(IIDPropensityModel):
#     def __init__(self, summary, *args, **kwargs):
#         self.summary = summary
#         self.model = LinearRegression(*args, **kwargs)

#     def fit(self, data):
#         Zs = np.array([
#             self.summary(data.Xs[data.G.N1(j)],
#                          data.G.sub(data.G.N1(j)))
#             for j in np.arange(data.n_node)])

#         model_fit = self.model.fit(Zs, data.Ts)
#         model_fit.sigma_ = np.sqrt(np.mean(
#             (data.Ts - model_fit.predict(Zs))**2
#         ))
#         return LinearIIDPropensityFit(self.summary, model_fit)
        
# class LinearIIDPropensityFit(IIDPropensityFit):
#     def __init__(self, summary, model_fit):
#         self.summary = summary
#         self.model_fit = model_fit
    
#     def predict_i(self, T, X_N1, G_N1):
#         Z = self.summary(X_N1, G_N1)
#         dZ = Z.shape[-1]
#         T_hat = self.model_fit.predict(Z.reshape([-1,dZ])).reshape(Z.shape[:-1])
#         return stats.norm.pdf((T - T_hat)/self.model_fit.sigma_)

#     def sample_i(self, n_sample, X_N1, G_N1):
#         Z = self.summary(X_N1, G_N1)
#         dZ = Z.shape[-1]
#         T_hat = self.model_fit.predict(Z.reshape([-1,dZ])).reshape(Z.shape[:-1])
#         return stats.norm.rvs(T_hat, self.model_fit.sigma_, size=(n_sample,)+T_hat.shape)




###
class LogisticIIDPropensityModel(IIDPropensityModel):
    def __init__(self, summary, *args, **kwargs):
        self.summary = summary
        self.model = LogisticRegression(penalty=None, fit_intercept=False,
                                        *args, **kwargs)

    def fit(self, data):
        Zs = np.array([
            self.summary(data.Xs[data.G.N1(j)],
                         data.G.sub(data.G.N1(j)))
            for j in np.arange(data.n_node)])
        Zs = np.concatenate([np.full(Zs.shape[:-1]+(1,), 1), Zs], -1)
        
        model_fit = self.model.fit(Zs, data.Ts)
        model_fit.Zs_ = Zs
        model_fit.Ts_ = data.Ts
        model_fit.residuals_ = data.Ts - model_fit.predict_proba(Zs)[...,-1]
        model_fit.var_ = (
            np.abs(model_fit.residuals_) * (1 - np.abs(model_fit.residuals_))
        )
        return LogisticIIDPropensityFit(self.summary, model_fit)

class LogisticIIDPropensityFit(IIDPropensityFit):
    def __init__(self, summary, model_fit):
        self.summary = summary
        self.model_fit = model_fit

    def predict_i(self, T, X_N1, G_N1):
        Z = self.summary(X_N1, G_N1)        
        Z = np.concatenate([np.full(Z.shape[:-1]+(1,), 1), Z], -1)
        dZ = Z.shape[-1]
        
        return np.abs(
            self.model_fit.predict_proba(Z.reshape([-1,dZ]))[:,0].reshape(Z.shape[:-1])
            - T
        )

    def predict_with_residual_i(self, T, X_N1, G_N1):
        Z = self.summary(X_N1, G_N1)
        Z = np.concatenate([np.full(Z.shape[:-1]+(1,), 1), Z], -1)
        dZ = Z.shape[-1]

        pi_i = np.abs(
            self.model_fit.predict_proba(Z.reshape([-1,dZ]))[:,0].reshape(Z.shape[:-1])
            - T
        )
        var_i = (1 - pi_i) * pi_i
        res_i = - (
            (((2*T - 1) * var_i)[...,None] * Z) 
            @ la.pinv((self.model_fit.Zs_.T * self.model_fit.var_)
                      @ self.model_fit.Zs_)
            @ (self.model_fit.Zs_.T * self.model_fit.residuals_)
        )
        
        return pi_i, res_i




###
class KernelIIDPropensityModel(IIDPropensityModel):
    def __init__(self, delta, lamda=None, ths=1e-3, *args, **kwargs):
        self.delta = delta
        self.lamda = lamda
        self.clip = - np.log(ths)

    def fit(self, data):
        return KernelIIDPropensityFit(self.delta, self.lamda, self.clip, data)

class KernelIIDPropensityFit(IIDPropensityFit):
    def __init__(self, delta, lamda, clip, data):
        self.delta = delta
        self.lamda = lamda
        self.clip = clip
        self.data = data

    def loo_cv(self, lamdas, i0s=None, n_process=1, tqdm=None, leave_tqdm=True):
        if tqdm is None:
            def tqdm(iterable, *args, **kwargs):
                return iterable

        if i0s is None:
            i0s = np.arange(self.data.n_node)
                
        # ks_cv = random.choice(np.arange(self.data.n_node), n_cv, replace=False)
        
        if n_process == 1:
            from itertools import starmap
            pis_cv = list(tqdm(starmap(self.loo_cv_k,
                ((k, lamdas, 1, None, False) 
                 for k in i0s)
            ), total=len(i0s), leave=leave_tqdm, desc='j', smoothing=0))
        elif n_process > 1:
            from multiprocessing import Pool
            with Pool(n_process) as p:   
                pis_cv = list(tqdm(p.istarmap(self.loo_cv_k,
                    ((k, lamdas, 1, None, False) 
                     for k in i0s)
                ), total=len(i0s), leave=leave_tqdm, desc='j', smoothing=0))

        return self.data.Ts[i0s], np.array(pis_cv).T

    def loo_cv_k(self, k, lamdas, n_process=1, tqdm=None, leave_tqdm=False):
        N1k = self.data.G.N1(k)
        N2k = self.data.G.N2(k)
        mk = np.delete(np.arange(self.data.n_node), N2k)
        
        # data_mk = Data(
        #     self.data.Ys[mk], self.data.Ts[mk], 
        #     self.data.Xs[mk], self.data.G.sub(mk)
        # )
        # fit_mk = KernelIIDPropensityModel(
        #     self.delta
        # ).fit(data_mk)
    
        # return fit_mk.predict_i(
        #     1, self.data.Xs[N1k], 
        #     self.data.G.sub(N1k), lamdas=lamdas
        # )

        Ds = np.stack(
            [self.delta(self.data.Xs[N1k], self.data.G.sub(N1k), 
                        self.data.Xs[self.data.G.N1(i)],
                        self.data.G.sub(self.data.G.N1(i)))
             for i in mk], -1
        )
        Ds = Ds - np.min(Ds, -1)[...,None]

        lamDs = lamdas.reshape(lamdas.shape+(1,)*Ds.ndim) * Ds
        ws = np.zeros(lamDs.shape)
        ws[lamDs < self.clip] = np.exp(- lamDs[lamDs < self.clip])

        return np.sum(self.data.Ts[mk] * ws, -1) / np.sum(ws, -1)

    def predict_i(self, T, X_N1, G_N1, lamdas=None, residual=False):
        if lamdas is None:
            lamdas = np.array(self.lamda)
        else:
            lamdas = np.array(lamdas)
            
        Ds = np.stack(
            [self.delta(X_N1, G_N1, 
                        self.data.Xs[self.data.G.N1(i)],
                        self.data.G.sub(self.data.G.N1(i)))
             for i in np.arange(self.data.n_node)], -1
        )
        Ds = Ds - np.min(Ds, -1)[...,None]

        lamDs = lamdas.reshape(lamdas.shape+(1,)*Ds.ndim) * Ds
        ws = np.zeros(lamDs.shape)
        ws[lamDs < self.clip] = np.exp(- lamDs[lamDs < self.clip])
        
        return np.abs(
            np.sum(self.data.Ts * ws, -1) / np.sum(ws, -1)
            - 1 + T
        )

    def predict_with_residual_i(self, T, X_N1, G_N1, lamdas=None):
        if lamdas is None:
            lamdas = np.array(self.lamda)
        else:
            lamdas = np.array(lamdas)
            
        Ds = np.stack(
            [self.delta(X_N1, G_N1, 
                        self.data.Xs[self.data.G.N1(i)],
                        self.data.G.sub(self.data.G.N1(i)))
             for i in np.arange(self.data.n_node)], -1
        )
        Ds = Ds - np.min(Ds, -1)[...,None]

        lamDs = lamdas.reshape(lamdas.shape+(1,)*Ds.ndim) * Ds
        ws = np.zeros(lamDs.shape)
        ws[lamDs < self.clip] = np.exp(- lamDs[lamDs < self.clip])

        pi_i = np.sum((np.array(T)[...,None] == self.data.Ts) * ws, -1) / np.sum(ws, -1)
        res_i = - (
            ((np.array(T)[...,None] == self.data.Ts) - pi_i[...,None]) * ws 
            / np.sum(ws, -1)[...,None]
        )
        
        return pi_i, res_i