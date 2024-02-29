import numpy as np
import pandas as pd
import scipy.sparse as sparse

class Data:
    def __init__(self, Ys, Ts, Xs, G):
        self.G = G
        self.n_node = G.n_node
        
        self.Ys = np.array(Ys)
        self.Ts = np.array(Ts)
        self.Xs = np.array(Xs)

class Graph:
    def __init__(self, Adj, Zs=None):
        self.Adj = np.array(Adj)
        
        if self.Adj.ndim != 2:
            raise('Adj is not a two-dimensional matrix')
        elif self.Adj.shape[0] != self.Adj.shape[1]:
            raise('Adj is not a square matrix')
        
        self.n_node = self.Adj.shape[0]
        self.Zs = Zs

    def sub(self, ids):
        if self.Zs is None:
            return Graph(self.Adj[np.ix_(ids,ids)])
        else:
            return Graph(self.Adj[np.ix_(ids,ids)], self.Zs[ids])

    def N1(self, i):
        return np.concatenate(
            [[i], np.nonzero(self.Adj[i])[0]]
        )

    def N2(self, i):
        return pd.unique(np.concatenate([
            np.concatenate([[j], np.nonzero(self.Adj[j])[0]])
            for j in self.N1(i)
        ]))

    def dist(self):
        return sparse.csgraph.floyd_warshall(self.Adj)