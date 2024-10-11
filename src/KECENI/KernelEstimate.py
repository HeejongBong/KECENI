import numpy as np
import numpy.random as random
import pandas as pd

class KernelEstimate:
    def __init__(self, fit, i0, T0, G0, lamdas, hs, Ds, xis, wms=None, wxms=None):
        self.fit = fit
        
        self.i0 = i0
        self.T0 = T0
        self.G0 = G0

        self.lamdas = np.array(lamdas)
        self.hs = np.array(hs)
        
        self.Ds = Ds
        self.xis = xis
        
        self.wms = wms
        self.wxms = wxms

        self.ws = np.exp(
            - self.lamdas.reshape(self.lamdas.shape+(1,)*(self.Ds.ndim-1))
            * self.Ds.reshape((self.fit.data.n_node,)+(1,)*self.lamdas.ndim+self.Ds.shape[1:])
        )

    def est(self):
        return np.sum(
            self.xis.reshape((self.fit.data.n_node,)+(1,)*self.lamdas.ndim+self.xis.shape[1:])
            * self.ws, 0
        ) / np.sum(self.ws, 0)
        
    def phis_del(self):
        return (
            self.xis.reshape((self.fit.data.n_node,)+(1,)*self.lamdas.ndim+self.xis.shape[1:])
            * self.ws - self.wxms - self.est() * (self.ws - self.wms)
        ) / np.sum(self.ws, 0)

    def phis_lapx(self):
        return (
            (self.xis.reshape((self.fit.data.n_node,)+(1,)*self.lamdas.ndim+self.xis.shape[1:])
             - self.est()) * (2 * self.ws - self.wms)
        ) / np.sum(self.ws, 0)