import numpy as np
import numpy.random as random
import pandas as pd

def parzen_kernel(x, bw=None, G=None, const=2, eps=0.05):
    x = np.array(x)

    if bw is None:
        bw = np.array(
            const * np.log(G.n_node) 
            / np.log(np.maximum(np.mean(np.sum(G.Adj, 0)), 1+eps))
        )
    else:
        bw = np.array(bw)
    
    z = x/bw.reshape(bw.shape+(1,)*x.ndim)
    w = np.zeros(z.shape)
    
    ind1 = (z <= 0.5)
    ind2 = (z > 0.5) & (z <= 1)
    
    w[ind1] = 1 - 6 * z[ind1]**2 * (1-z[ind1])
    w[ind2] = 2 * (1-z[ind2])**3
    
    return w

def box_kernel(x, bw=None, G=None):
    x = np.array(x)

    if bw is None:
        bw = np.array(
            const * np.log(G.n_node) 
            / np.log(np.maximum(np.mean(np.sum(G.Adj, 0)), 1+eps))
        )
    else:
        bw = np.array(bw)
    
    z = x/bw.reshape(bw.shape+(1,)*x.ndim)
    w = np.zeros(z.shape)
    
    ind1 = (z <= 1)
    
    w[ind1] = 1
    
    return w



# inference for sdw

def mse_hac_sdw(phis, G, hac_kernel=parzen_kernel, abs=False, **kwargs):
    if abs:
        return np.sum(np.abs(phis) * np.tensordot(
            hac_kernel(G.dist, G=G, **kwargs), np.abs(phis), axes=(-1,0)
        ), -phis.ndim)
    else:
        return np.sum(phis * np.tensordot(
            hac_kernel(G.dist, G=G, **kwargs), phis, axes=(-1,0)
        ), -phis.ndim)

def ste_hac_sdw(phis, G, hac_kernel=parzen_kernel, abs=False, **kwargs):
    return np.sqrt(mse_hac_sdw(phis, G, abs=abs, hac_kernel=hac_kernel, **kwargs))

def bbb_sdw(phis, G, hops=1, n_bst=1000, tqdm=None, level_tqdm=0):
    if tqdm is None:
        def tqdm(iterable, *args, **kwargs):
            return iterable

    hops = np.array(hops)
    Ks = G.n_node / np.mean(np.sum(G.dist <= hops[...,None,None], -1),-1)
    Ks_all = G.n_node / np.mean(np.sum(
        G.dist <= np.arange(np.max(hops).astype(int)+1)[1:,None,None], -1
    ), -1)

    phis_bst = np.zeros((n_bst,)+hops.shape+phis.shape[1:])
    for i_bst in tqdm(range(n_bst), smoothing=0, desc='bst', leave=None, position=level_tqdm):
        id_smp = np.random.choice(G.n_node, np.max(Ks.astype(int)), replace=True)
        bs_bst = np.logical_and(
            G.dist[id_smp] <= hops[...,None,None], 
            np.sum(
                Ks_all[:,None].astype(int) > np.arange(np.max(Ks).astype(int)), 0
            )[:,None] >= hops[...,None,None]
        )
        phis_bst[i_bst] = np.sum(
            np.sum(bs_bst, -2).reshape(hops.shape+(G.n_node,)+(1,)*(phis.ndim-1))
            * phis, hops.ndim
        ) * (Ks / Ks.astype(int)).reshape(hops.shape+(1,)*(phis.ndim-1))

    return phis_bst

def mse_bbb_sdw(phis, G, hops=1, n_bst=1000, tqdm=None, level_tqdm=0):
    if tqdm is None:
        def tqdm(iterable, *args, **kwargs):
            return iterable

    phis_bst = bbb_sdw(phis, G, hops, n_bst, tqdm, level_tqdm)

    return np.var(phis_bst, 0)

def ste_bbb_sdw(phis, G, hops=1, n_bst=1000, tqdm=None, level_tqdm=0):
    return np.sqrt(mse_bbb_sdw(phis, G, hops, n_bst, tqdm, level_tqdm))




# inference for bst

def mse_hac_bst(phis, G, hac_kernel=parzen_kernel, abs=False, **kwargs):
    return (
        np.mean(mse_hac_sdw(phis, G, hac_kernel, abs, **kwargs), -phis.ndim+1) 
        + np.var(np.sum(phis, 0), 0)
    )

def ste_hac_bst(phis, G, hac_kernel=parzen_kernel, abs=False, **kwargs):
    return np.sqrt(mse_hac_bst(phis, G, abs=abs, hac_kernel=hac_kernel, **kwargs))

def bbb_bst(phis, G, hops=1, tqdm=None, level_tqdm=0):
    if tqdm is None:
        def tqdm(iterable, *args, **kwargs):
            return iterable
        
    n_bst = phis.shape[1]

    hops = np.array(hops)
    Ks = G.n_node / np.mean(np.sum(G.dist <= hops[...,None,None], -1),-1)
    Ks_all = G.n_node / np.mean(np.sum(
        G.dist <= np.arange(np.max(hops).astype(int)+1)[1:,None,None], -1
    ), -1)

    phis_bst = np.zeros((n_bst,)+hops.shape+phis.shape[2:])
    for i_bst in tqdm(range(n_bst), smoothing=0, desc='bst'):
        id_smp = np.random.choice(G.n_node, np.max(Ks.astype(int)), replace=True)
        bs_bst = np.logical_and(
            G.dist[id_smp] <= hops[...,None,None], 
            np.sum(
                Ks_all[:,None].astype(int) > np.arange(np.max(Ks).astype(int)), 0
            )[:,None] >= hops[...,None,None]
        )
        phis_bst[i_bst] = np.sum(
            np.sum(bs_bst, -2).reshape(hops.shape+(G.n_node,)+(1,)*(phis.ndim-2))
            * phis[:,i_bst], hops.ndim
        ) * (Ks / Ks.astype(int)).reshape(hops.shape+(1,)*(phis.ndim-2))

    return phis_bst

def mse_bbb_bst(phis, G, hops=1, tqdm=None, level_tqdm=0):
    if tqdm is None:
        def tqdm(iterable, *args, **kwargs):
            return iterable

    phis_bst = bbb_bst(phis, G, hops, tqdm, level_tqdm)

    return np.var(phis_bst, 0)

def ste_bbb_bst(phis, G, hops=1, tqdm=None, level_tqdm=0):
    return np.sqrt(mse_bbb_bst(phis, G, hops, tqdm, level_tqdm))