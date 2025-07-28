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
n_X = 50

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
    
    ## inference
    Hs_nu = result_fit.Hs_nu(n_X=n_X, tqdm=tqdm)
    Hs_px = result_fit.Hs_px(n_X=n_X, tqdm=tqdm)
    
    np.savez('%s/inf_%.3d.npz'%(save_dir,task_id), Hs_nu=Hs_nu, Hs_px=Hs_px)

if __name__ == "__main__":
    # Get task ID from SLURM_ARRAY_TASK_ID
    if len(sys.argv) != 2:
        raise ValueError("Please provide the SLURM_ARRAY_TASK_ID as an argument.")
    task_id = int(sys.argv[1])

    # Run the simulation for this task ID
    run_task(task_id)
