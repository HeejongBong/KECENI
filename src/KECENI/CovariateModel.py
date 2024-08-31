import numpy as np
import numpy.random as random
import pandas as pd



###
class CovariateModel:
    def __init__(self):
        pass
        
    def fit(self, data):
        return Covariatefit()

class CovariateFit:
    def __init__(self):
        pass

    def sample(self, n_sample, N2, G):
        n2 = N2.shape[0]
        return np.zeros((n_sample, n2))



###
class FittedCovariateModel:
    def __init__(self, rX):
        self.rX = rX

    def fit(self, data):
        return FittedCovariateFit(self.rX)

class FittedCovariateFit:
    def __init__(self, rX):
        self.rX = rX

    def sample(self, n_sample, N2, G):
        return self.rX(n_sample, N2, G)



###
class IIDCovariateModel(CovariateModel):
    def __init__(self):
        pass
        
    def fit(self, data):
        return IIDCovariateFit(data.Xs)

class IIDCovariateFit:
    def __init__(self, Xs):        
        self.Xs = np.array(Xs)
        self.dX = Xs.shape[-1]
        self.n_node = Xs.shape[0]

    def sample(self, n_sample, N2, G):
        n2 = N2.shape[0]
        rind = random.choice(np.arange(self.n_node), size=(n_sample, n2))
        return self.Xs[rind.flatten()].reshape(rind.shape + (self.dX,))



###
class CommunityCovariateModel(CovariateModel):
    def __init__(self):
        pass
        
    def fit(self, data):
        return CommunityCovariateFit(data.Xs, data.G)

class CommunityCovariateFit:
    def __init__(self, Xs, G):        
        self.Xs = np.array(Xs)
        self.dX = Xs.shape[-1]
        self.comm_dict = pd.Series(G.Zs[:,0]).groupby(G.Zs[:,0]).groups

    def sample(self, n_sample, N2, G):
        n2 = N2.shape[0]
        G_N2 = G.sub(N2)
        comm_dict_N2 = pd.Series(G_N2.Zs[:,0]).groupby(G_N2.Zs[:,0]).groups

        Xs_sample = np.zeros((n_sample, n2, self.dX))
        for k, v in comm_dict_N2.items():
            Xs_sample[:,v,:] = self.Xs[np.random.choice(self.comm_dict[k], (n_sample, len(v)))]
        return Xs_sample