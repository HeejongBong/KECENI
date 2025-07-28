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

from hyperparams import delta, HyperAlpha

n_process = 4
lamdas = np.linspace(0, 20, 21)[1:]
hops = np.array([5,6,7,8])

# from your_module import task_AIPW, KECENI, KernelEstimate  # Adjust import paths as needed

def run_task(task_id):
    save_dir = 'result_misspi/result_%.3d.npz'%task_id
    
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
    hyper = HyperAlpha(0, 1)

    keceni_model = KECENI.Model(
        LinearRegressionModel(hyper.summary_mu),
        LogisticIIDPropensityModel(hyper.summary_pi),
        IIDCovariateModel(),
        delta
    )
    keceni_fit = keceni_model.fit(data, tqdm=tqdm)

    ## cross-validation
    result_cv = keceni_fit.cv(
        tqdm=tqdm
    )
    xs_cv, xhs_cv = result_cv.xs_xhs(lamdas)
    id_cv = np.argmin(np.mean((xs_cv-xhs_cv)**2, -1))
    
    ## G-computation
    result_G = keceni_fit.G_estimate(i0s, T0s, n_X=1000)
    YG_0, YG_1 = np.moveaxis(result_G, -1, 0)
    YG_d = YG_1 - YG_0
    
    ## kernel AIPW
    result_AIPW = keceni_fit.kernel_AIPW(
        i0s, T0s, tqdm=tqdm
    )
    YDR_0, YDR_1 = np.moveaxis(result_AIPW.est(lamdas), -1, 0)
    YDR_d = YDR_1 - YDR_0
    
    # inference
    ## phis and Hs
    phis_eif = result_AIPW.phis_eif(lamdas)
    # wH_nu = result_AIPW.wH_nu(lamdas, tqdm=tqdm)
    # wH_px = result_AIPW.wH_px(lamdas, tqdm=tqdm)
    wH_nu = result_AIPW.Hs_nu(lamdas, n_X=100, tqdm=tqdm)
    wH_px = result_AIPW.Hs_px(lamdas, n_sample=10, tqdm=tqdm)
    
    phis_dr = phis_eif + wH_nu + wH_px
    
    ## inference
    _ = G.get_dist()
    ste_bbb_dr = inf.ste_bbb_sdw(phis_dr, G, hops, n_bst=100)
    ste_hac_dr = inf.ste_hac_sdw(phis_dr, G, inf.box_kernel, bw=hops)
    
    sdd_bbb_dr = inf.ste_bbb_sdw(phis_dr[...,0] - phis_dr[...,1], G, hops, n_bst=100)
    sdd_hac_dr = inf.ste_hac_sdw(phis_dr[...,0] - phis_dr[...,1], G, inf.box_kernel, bw=hops)
    
    # save results
    np.savez(save_dir, YG_0 = YG_0, YG_1=YG_1, YG_d=YG_d,
             xs_cv=xs_cv, xhs_cv=xhs_cv, id_cv=id_cv,
             YDR_0=YDR_0, YDR_1=YDR_1, YDR_d=YDR_d,
             sd0_bbb_dr=ste_bbb_dr[...,0], sd1_bbb_dr=ste_bbb_dr[...,1],
             sdd_bbb_dr=sdd_bbb_dr,
             sd0_hac_dr=ste_hac_dr[...,0], sd1_hac_dr=ste_hac_dr[...,1],
             sdd_hac_dr=sdd_hac_dr)
             

if __name__ == "__main__":
    # Get task ID from SLURM_ARRAY_TASK_ID
    if len(sys.argv) != 2:
        raise ValueError("Please provide the SLURM_ARRAY_TASK_ID as an argument.")
    task_id = int(sys.argv[1])

    # Run the simulation for this task ID
    run_task(task_id)

