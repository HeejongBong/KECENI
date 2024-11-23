import numpy as np

class IT_broadcaster:
    def __init__(self, i0s, T0s):
        self.n_node = T0s.shape[-1]
        self.Tfs = T0s.reshape([-1, self.n_node])
        self.Txs = np.arange(self.Tfs.shape[0]).reshape(T0s.shape[:-1])
        self.b = np.broadcast(i0s, self.Txs)
        
    def __iter__(self):
        for i0, Tx in self.b:
            yield i0, self.Tfs[Tx]