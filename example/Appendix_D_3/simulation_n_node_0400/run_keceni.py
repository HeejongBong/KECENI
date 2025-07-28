import os, importlib, sys, time

import numpy as np
import scipy.sparse as sparse
import pandas as pd
import pyarrow
from tqdm import tqdm

import KECENI
import KECENI.Inference as inf

from KECENI.RegressionModel import LinearRegressionModel
from KECENI.PropensityModel import LogisticIIDPropensityModel
from KECENI.CovariateModel import IIDCovariateModel

from hyperparams import delta, summary_mu, summary_pi

lamdas = np.linspace(0, 20, 21)[1:]
hops = np.array([5,6,7,8])

n_sim = 80
n_X = 1000
n_bst = 120

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
    i0 = 0
    T0s_0 = Ts.copy(); T0s_0[:] = 0
    T0s_1 = Ts.copy(); T0s_1[:] = 1
    
    i0s = np.array([i0])
    T0s = np.array([T0s_0, T0s_1])
    
    # estimation
    keceni_model = KECENI.Model(
        LinearRegressionModel(summary_mu),
        LogisticIIDPropensityModel(summary_pi),
        IIDCovariateModel(bal=False),
        delta,
    )
    keceni_fit = keceni_model.fit(data, n_X=n_X, tqdm=tqdm)
    
    ## G-computation
    result_G = keceni_fit.G_estimate(i0s, T0s, n_X=n_X)
    YG_0, YG_1 = np.moveaxis(result_G, -1, 0)
    YG_d = YG_1 - YG_0
    
    ## cross-validation
    result_cv = keceni_fit.cv(tqdm=tqdm)
    xs_cv, xhs_cv = result_cv.xs_xhs(lamdas)
    id_cv = np.argmin(np.mean((xs_cv-xhs_cv)**2, -1))
    
    ## KECENI
    result_AIPW = keceni_fit.kernel_AIPW(
        i0s, T0s, tqdm=tqdm
    )
    YDR_0, YDR_1 = np.moveaxis(result_AIPW.est(lamdas), -1, 0)
    YDR_d = YDR_1 - YDR_0
    
    # save results
    np.savez(save_dir, YG_0 = YG_0, YG_1=YG_1, YG_d=YG_d,
             xs_cv=xs_cv, xhs_cv=xhs_cv, id_cv=id_cv,
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

