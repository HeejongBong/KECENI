import os, importlib, sys, time

import numpy as np
import scipy.sparse as sparse
import pandas as pd
import pyarrow
from tqdm import tqdm

import KECENI

from KECENI.RegressionModel import KernelRegressionModel
from KECENI.PropensityModel import KernelIIDPropensityModel
from KECENI.CovariateModel import IIDCovariateModel, IIDCovariateFit

from hyperparams import delta_mu, delta_pi, delta

lamda_mu = 3.0
lamda_pi = 4.0
lamdas = 4.0
n_X = 50

# from your_module import task_AIPW, KECENI, KernelEstimate  # Adjust import paths as needed

def run_task(task_id):   
    save_dir = 'results'
    
    # load data
    data_network = pd.read_feather('data/network.feather')
    data_feature = pd.read_feather('data/feature.feather')
    
    n_node = len(data_feature)
    n_edge = len(data_network)

    Adj = sparse.csr_matrix((
        np.full(len(data_network), True), 
        (np.array(data_network.row)-1, np.array(data_network.col)-1)
    ), shape=(n_node,n_node)).toarray()
    Zs = data_feature.village.values[:,None]
    G = KECENI.Graph(Adj[:n_node,:n_node], Zs=Zs[:n_node])
    
    idx_cov = np.array([2, 3, 4, 5, 7])
    
    Ys = data_feature.loan.values.astype(float)[:n_node]
    Ts = data_feature.shg.values.astype(float)[:n_node]
    Xs = (
        (data_feature.iloc[:n_node,idx_cov].values 
         - np.mean(data_feature.iloc[:n_node,idx_cov].values, 0))
        / np.array([100., 14., 1., 1., 1.])
    )
    
    data = KECENI.Data(Ys, Ts, Xs, G)
    
    # set division    
    js = np.arange(n_node)[task_id*90:(task_id+1)*90]
    
    # load fit
    keceni_model = KECENI.Model(
        KernelRegressionModel(delta_mu, lamda=lamda_mu, ths=1e-7),
        KernelIIDPropensityModel(delta_pi, lamda=lamda_pi, ths=1e-7),
        IIDCovariateModel(),
        delta
    )
    
    result_fit = KECENI.Fit(
        keceni_model, data,
        **np.load('%s/fit_%.3d.npz'%(save_dir,task_id))
    )
    
    # counterfactual of interest
    i0s = np.arange(n_node)[::20]
    T0s_0 = np.repeat(Ts[None,:], len(i0s), 0); T0s_0[np.arange(len(i0s)), i0s] = 0
    T0s_1 = np.repeat(Ts[None,:], len(i0s), 0); T0s_1[np.arange(len(i0s)), i0s] = 1
    T0s = np.array([T0s_0, T0s_1])
    
    # kernel AIPW
    result_AIPW = result_fit.kernel_AIPW(
        i0s, T0s, None, n_process=1, tqdm=tqdm, level_tqdm=0
    )
    
    np.savez('%s/AIPW_direct_%.3d.npz'%(save_dir,task_id), Ds=result_AIPW.Ds)
             

if __name__ == "__main__":
    # Get task ID from SLURM_ARRAY_TASK_ID
    if len(sys.argv) != 2:
        raise ValueError("Please provide the SLURM_ARRAY_TASK_ID as an argument.")
    task_id = int(sys.argv[1])

    # Run the simulation for this task ID
    run_task(task_id)
