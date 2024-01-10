import numpy as np
import pandas as pd

class Data:
    def __init__(self, Ys, Ts, Xs, G):
        self.G = np.array(G)
        self.n_node = G.shape[0]
        self.N1s = [np.concatenate(
            [[i], np.nonzero(Gi)[0]]
        ) for i, Gi in enumerate(self.G)]
        self.N2s = [pd.unique(np.concatenate(
            [self.N1s[i] for i in N1j]
        )) for N1j in self.N1s]
        
        self.Ys = np.array(Ys)
        self.Ts = np.array(Ts)
        self.Xs = np.array(Xs)