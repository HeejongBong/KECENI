import numpy as np
import numpy.random as random
import pandas as pd

class KernelEstimate:
    def __init__(self, fit, i0s, T0s, G0, lamdas, Ds):
        self.fit = fit
        
        self.i0s = i0s
        self.T0s = T0s
        self.G0 = G0
        
        self.lamdas = np.array(lamdas)
        self.Ds = Ds
        self.ws = np.exp(
            - lamdas.reshape(lamdas.shape+(1,)*(Ds.ndim-1))
            * Ds.reshape((self.fit.data.n_node,)+(1,)*lamdas.ndim+Ds.shape[1:])
        )

    def est(self):
        return np.sum(
            self.fit.xis.reshape((self.fit.data.n_node,)+(1,)*(self.lamdas.ndim+self.Ds.ndim-1))
            * self.ws, 0
        ) / np.sum(self.ws, 0)

    def phis_dr(self):
        return (
            (self.fit.xis.reshape((self.fit.data.n_node,)+(1,)*self.lamdas.ndim+self.xis.shape[1:])
             - self.est()[...,None]) * self.ws[...,None]
        ) / np.sum(self.ws, 0)[...,None] + self.offsets