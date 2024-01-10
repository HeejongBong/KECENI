import numpy as np
import numpy.random as random



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