import os, importlib, sys, time

import numpy as np
import scipy.sparse as sparse
import pandas as pd
import pyarrow

from tqdm import tqdm
import itertools

import KECENI

from KECENI.RegressionModel import LinearRegressionModel
from KECENI.PropensityModel import LogisticIIDPropensityModel
from KECENI.CovariateModel import IIDCovariateModel

from hyperparams import delta, HyperAlpha

lamdas = np.linspace(0, 20, 21)[1:]

alphas_mu = np.linspace(0, 1, 11)
alphas_pi = np.linspace(0, 1, 11)

n_sim = 80
n_X = 1000

def run_task(task_id):
    save_dir = 'result/result_%.3d.npz'%task_id
    
    # load data
    data_network = pd.read_feather('data/network.feather')
    data_latent = pd.read_feather('data/latent.feather')
    data_feature = pd.read_feather('data/feature_%.3d.feather'%task_id)
    
    n_node = len(data_latent)
    Adj = sparse.csr_matrix((
        np.full(len(data_network), True), 
        (np.array(data_network.row)-1, np.array(data_network.col)-1)
    ), shape=(n_node,n_node)).toarray()
    G = KECENI.Graph(Adj)

    Ys = data_feature.iloc[:,6].values
    Ts = data_feature.iloc[:,4].values
    Xs = data_feature.iloc[:,0:3].values
    
    data = KECENI.Data(Ys, Ts, Xs, G)
    
    # counterfactual of interest
    i0 = 17
    T0s_1 = np.zeros(n_node); T0s_1[G.N1(i0)[::2]] = 1
    T0s_0 = T0s_1.copy(); T0s_0[i0] = 0
    
    i0s = np.array([i0])
    T0s = np.array([T0s_0, T0s_1])
    
    # estimation
    mse_cv = np.zeros(alphas_mu.shape + alphas_pi.shape + lamdas.shape)
    id_cv = np.zeros(alphas_mu.shape + alphas_pi.shape, dtype=int)
    result_G = np.zeros(alphas_mu.shape + alphas_pi.shape + (2,))
    result_AIPW = np.zeros(alphas_mu.shape + alphas_pi.shape + lamdas.shape + (2,))

    for (i, alpha_mu), (j, alpha_pi) \
    in tqdm(
        itertools.product(enumerate(alphas_mu), enumerate(alphas_pi)), 
        total=len(alphas_mu)*len(alphas_pi),
        desc='alpha_mu, alpha_pi', position=0, leave=True
    ):
        hyper_ij = HyperAlpha(alpha_mu, alpha_pi)
        
        keceni_model = KECENI.Model(
            LinearRegressionModel(hyper_ij.summary_mu),
            LogisticIIDPropensityModel(hyper_ij.summary_pi),
            IIDCovariateModel(),
            delta
        )
        keceni_fit = keceni_model.fit(data, n_X=n_X)
        
        ## G-computation
        result_G[i,j] = keceni_fit.G_estimate(i0s, T0s, n_X=n_X)
        
        ## cross-validation
        result_cv = keceni_fit.cv()
        xs_cv, xhs_cv = result_cv.xs_xhs(lamdas)

        mse_cv[i,j] = np.mean((xs_cv-xhs_cv)**2, -1)
        id_cv[i,j] = np.argmin(np.mean((xs_cv-xhs_cv)**2, -1))
        
        ## KECENI
        result_AIPW[i,j] = keceni_fit.kernel_AIPW(
            i0s, T0s
        ).est(lamdas)
    
    # save results
    np.savez(save_dir, mse_cv=mse_cv, id_cv=id_cv,
             result_G=result_G, result_AIPW=result_AIPW)


if __name__ == "__main__":
    # Get task ID from SLURM_ARRAY_TASK_ID
    if len(sys.argv) < 2:
        for i in range(n_sim):
            run_task(i)
    else:
        task_id = int(sys.argv[1])

        # Run the simulation for this task ID
        run_task(task_id)

