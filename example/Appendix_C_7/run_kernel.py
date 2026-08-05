import os, importlib, sys, time

import numpy as np
import scipy.sparse as sparse
import pandas as pd
import pyarrow
from tqdm import tqdm

import KECENI

from KECENI.RegressionModel import KernelRegressionModel
from KECENI.PropensityModel import KernelIIDPropensityModel
from KECENI.CovariateModel import IIDCovariateModel

from hyperparams import delta, delta_mu, delta_pi

lamdas = np.linspace(0, 20, 21)[1:]

n_sim = 40
n_X = 100
n_process = 32

def run_task(task_id):
    save_dir = 'result_kernel/result_%.3d.npz'%task_id
    
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
    
    Ys = data_feature.iloc[:,9].values
    Ts = data_feature.iloc[:,7].values
    Xs = data_feature.iloc[:,3:6].values
    
    data = KECENI.Data(Ys, Ts, Xs, G)
    
    # counterfactual of interest
    i0s = np.arange(0,n_node,10)
    T0s_0 = Ts.copy(); T0s_0[:] = 0
    T0s_1 = Ts.copy(); T0s_1[:] = 1
    
    T0s = np.stack([T0s_0, T0s_1], 0)
    
    # estimation
    ## cross-validation for mu
    mu_fit = KernelRegressionModel(delta_mu).fit(data)
    _, mus_cv = mu_fit.loo_cv(lamdas, tqdm=tqdm, n_process=n_process)
    id_mcv = np.argmin(np.mean((mus_cv-Ys)**2, -1))
    lamda_mu = lamdas[id_mcv]

    ## cross-validation for pi
    pi_fit = KernelIIDPropensityModel(delta_pi).fit(data)
    _, pis_cv = pi_fit.loo_cv(lamdas, tqdm=tqdm, n_process=n_process)
    id_pcv = np.argmin(np.mean(-np.log(np.abs(pis_cv + Ts - 1)), -1))
    lamda_pi = lamdas[id_pcv]

    ## cross-validation for KECENI
    keceni_model = KECENI.Model(
        KernelRegressionModel(delta_mu, lamda=lamda_mu),
        KernelIIDPropensityModel(delta_pi, lamda=lamda_pi),
        IIDCovariateModel(),
        delta
    )
    keceni_fit = keceni_model.fit(data, n_X=n_X, tqdm=tqdm)

    result_cv = keceni_fit.cv(
        i0s=i0s, tqdm=tqdm
    )
    xs_cv, xhs_cv = result_cv.xs_xhs(lamdas)
    id_cv = np.argmin(np.mean((xs_cv-xhs_cv)**2, -1))
    
    ## G-computation
    result_G = keceni_fit.G_estimate(
        i0s[:,None], T0s, n_X=1000, tqdm=tqdm
    )
    YG_0, YG_1 = np.moveaxis(result_G, -1, 0)
    YG_d = YG_1 - YG_0
    
    ## KECENI
    result_AIPW = keceni_fit.kernel_AIPW(
        i0s[:,None], T0s, tqdm=tqdm
    )
    YDR_0, YDR_1 = np.moveaxis(result_AIPW.est(lamdas), -1, 0)
    YDR_d = YDR_1 - YDR_0
    
    # save results
    np.savez(save_dir, YG_0 = YG_0, YG_1=YG_1, YG_d=YG_d,
             xs_cv=xs_cv, xhs_cv=xhs_cv, 
             id_mcv=id_mcv, id_pcv=id_pcv, id_cv=id_cv,
             YDR_0=YDR_0, YDR_1=YDR_1, YDR_d=YDR_d)

if __name__ == "__main__":
    # Get task ID from SLURM_ARRAY_TASK_ID
    if len(sys.argv) < 2:
        for i in range(n_sim):
            run_task(i)
    else:
        task_id = int(sys.argv[1])

        # Run the simulation for this task ID
        run_task(task_id)

