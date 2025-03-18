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
    
    def predict(self, f, n_sample, N2, G):
        Xs_bst = self.sample(n_sample, N2, G)
        return np.mean(f(Xs_bst), -1)
    
    def H(self, f, n_sample, N2, G):
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
        
    def H(self, f, n_sample, N2, G):
        Ehf = self.predict(f, n_sample, N2, G)
        return np.zeros(Ehf.shape + (1,))

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
        
#     def H(self, f, n_sample, N2, G, n_bst=None):
#         if n_bst is None:
#             n_bst = self.n_node
            
#         is_bst = np.random.choice(self.n_node, n_bst, False)
#         Xs_bst = self.sample(n_sample, N2, G)
#         shape_f = f(Xs_bst[0]).shape[:-1]
        
# #         H_tmp = np.stack([
# #             sum((
# #                 np.mean(f(
# #                     np.insert(
# #                         np.delete(Xs_bst, k, -2), 
# #                         k, self.Xs[j,None,:], -2
# #                     )
# #                 ), -1)
# #                 for k in range(N2.size)
# #             ))
# #             for j in is_bst
# #         ], -1) / np.sqrt(n_bst * self.n_node)
        
#         H_tmp = sum((
#             np.mean(f(
#                 np.insert(
#                     np.delete(np.repeat(Xs_bst[None,...], n_bst, 0), k, -2), 
#                     k, self.Xs[is_bst,None,:], -2
#                 ).reshape((-1, N2.size, self.Xs.shape[-1]))
#             ).reshape(shape_f + (n_bst, n_sample)), -1)
#             for k in np.arange(N2.size)
#         )) / n_bst # np.sqrt(n_bst * self.n_node)
        
# #         H = np.sum(np.mean(fs_swap_k.reshape(
# #             fs_swap_k.shape[:-2] + (self.n_node, n_sample) + (-1,)
# #         ), -2), -1) / self.n_node
        
#         H = np.zeros(H_tmp.shape[:-1] + (self.n_node,))
#         H[...,is_bst] = H_tmp - np.mean(H_tmp, -1)[...,None]
               
#         return H

    def H(self, f, n_sample, N2, G, n_bst=None):
        if n_bst is None:
            n_bst = self.n_node
            
        rind = np.random.choice(np.arange(self.n_node), size=(n_sample, N2.size))
        Xs_bst = self.Xs[rind.flatten()].reshape(rind.shape + (self.dX,))
        fs_bst = f(Xs_bst)
        
        n_cols = rind.shape[-1]
        A_flat = rind.ravel()  # shape: (n_rows * n_cols,)

        # 2) Replicate w so it lines up with A_flat
        #    For each row i, repeat w[i] across its n_cols entries
        w_repeated = np.repeat(fs_bst - np.mean(fs_bst), n_cols)  # shape: (n_rows * n_cols,)

        # 3) Identify unique values and get "inverse indices" that map each element to its group
        unique_vals, inv_idx = np.unique(A_flat, return_inverse=True)
        # Now inv_idx is a 1D array of the same length as A_flat,
        # telling you which unique_vals index each A_flat element belongs to.

        # 4a) Count how many times each unique value appears
        counts = np.bincount(inv_idx)
        # 4b) Sum the weights for each unique value
        weight_sums = np.bincount(inv_idx, weights=w_repeated)

        # 5) Build the output dictionary
        xi = np.zeros((self.n_node,))
#         for val, cnt, sm in zip(unique_vals, counts, weight_sums):
#             H[val] = sm / n_sample
        xi[unique_vals] = weight_sums / counts
    
        H = (xi - np.mean(xi, -1)[...,None]) * N2.size / self.n_node
        
#         H = np.sum(np.mean(fs_swap_k.reshape(
#             fs_swap_k.shape[:-2] + (self.n_node, n_sample) + (-1,)
#         ), -2), -1) / self.n_node
               
        return H

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
        
    def H(self, f, n_sample, N2, G):
        pass

    def sample(self, n_sample, N2, G):
        n2 = N2.shape[0]
        G_N2 = G.sub(N2)
        comm_dict_N2 = pd.Series(G_N2.Zs[:,0]).groupby(G_N2.Zs[:,0]).groups

        Xs_sample = np.zeros((n_sample, n2, self.dX))
        for k, v in comm_dict_N2.items():
            Xs_sample[:,v,:] = self.Xs[np.random.choice(self.comm_dict[k], (n_sample, len(v)))]
        return Xs_sample