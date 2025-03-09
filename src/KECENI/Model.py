from .Fit import Fit

class Model:
    def __init__(self, mu_model, pi_model, cov_model, delta):
        self.mu_model = mu_model
        self.pi_model = pi_model
        self.cov_model = cov_model
        self.delta = delta
        
    def fit(self, data=None, **kwargs):
        return Fit(
            self, data, **kwargs
        )    