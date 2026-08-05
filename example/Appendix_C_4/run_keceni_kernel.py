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

from KECENI.KernelEstimate import box_kernel, exponential_kernel, triangular_kernel, gaussian_kernel

from hyperparams import delta, summary_mu, summary_pi

lamdas = np.linspace(0, 20, 21)[1:]

n_sim = 80
n_X = 1000
n_bst = 120

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

    xs_exp, xhs_exp = result_cv.xs_xhs(lamdas, kernel=exponential_kernel)
    xs_box, xhs_box = result_cv.xs_xhs(lamdas, kernel=box_kernel)
    xs_triang, xhs_triang = result_cv.xs_xhs(lamdas, kernel=triangular_kernel)
    xs_gaussian, xhs_gaussian = result_cv.xs_xhs(lamdas, kernel=gaussian_kernel)

    id_exp = np.argmin(np.mean((xs_exp-xhs_exp)**2, -1))
    id_box = np.argmin(np.mean((xs_box-xhs_box)**2, -1))
    id_triang = np.argmin(np.mean((xs_triang-xhs_triang)**2, -1))
    id_gaussian = np.argmin(np.mean((xs_gaussian-xhs_gaussian)**2, -1))
    
    ## KECENI
    result_AIPW = keceni_fit.kernel_AIPW(
        i0s, T0s, tqdm=tqdm
    )
    YDR_0_exp, YDR_1_exp = np.moveaxis(result_AIPW.est(lamdas, exponential_kernel), -1, 0)
    YDR_d_exp = YDR_1_exp - YDR_0_exp

    YDR_0_box, YDR_1_box = np.moveaxis(result_AIPW.est(lamdas, box_kernel), -1, 0)
    YDR_d_box = YDR_1_box - YDR_0_box

    YDR_0_triang, YDR_1_triang = np.moveaxis(result_AIPW.est(lamdas, triangular_kernel), -1, 0)
    YDR_d_triang = YDR_1_triang - YDR_0_triang

    YDR_0_gaussian, YDR_1_gaussian = np.moveaxis(result_AIPW.est(lamdas, gaussian_kernel), -1, 0)
    YDR_d_gaussian = YDR_1_gaussian - YDR_0_gaussian
    
    # save results
    np.savez(save_dir, YG_0 = YG_0, YG_1=YG_1, YG_d=YG_d,
             xs_exp=xs_exp, xhs_exp=xhs_exp, id_exp=id_exp,
             xs_box=xs_box, xhs_box=xhs_box, id_box=id_box,
             xs_triang=xs_triang, xhs_triang=xhs_triang, id_triang=id_triang,
             xs_gaussian=xs_gaussian, xhs_gaussian=xhs_gaussian, id_gaussian=id_gaussian,
             YDR_0_exp=YDR_0_exp, YDR_1_exp=YDR_1_exp, YDR_d_exp=YDR_d_exp,
             YDR_0_box=YDR_0_box, YDR_1_box=YDR_1_box, YDR_d_box=YDR_d_box,
             YDR_0_triang=YDR_0_triang, YDR_1_triang=YDR_1_triang, YDR_d_triang=YDR_d_triang,
             YDR_0_gaussian=YDR_0_gaussian, YDR_1_gaussian=YDR_1_gaussian, YDR_d_gaussian=YDR_d_gaussian)


if __name__ == "__main__":
    # Get task ID from SLURM_ARRAY_TASK_ID
    if len(sys.argv) < 2:
        for i in range(n_sim):
            run_task(i)
    else:
        task_id = int(sys.argv[1])

        # Run the simulation for this task ID
        run_task(task_id)

