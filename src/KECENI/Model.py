from .Fit import Fit

class Model:
    def __init__(self, mu_model, pi_model, cov_model, delta, nu_method='ksm'):
        self.mu_model = mu_model
        self.pi_model = pi_model
        self.cov_model = cov_model
        
        self.delta = delta
        self.nu_method = nu_method

    def fit(self, data, **kwargs):
        return Fit(
            self, data, self.nu_method, **kwargs
        )    