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

from KECENI.KernelEstimate import exponential_kernel

from hyperparams import avg_mu, avg_pi, avg, wavg_mu, wavg_pi, wavg, GW, L1

lamdas = np.linspace(0, 20, 21)[1:]

n_sim = 80
n_X = 1000
n_bst = 120

def run_task(task_id):
    save_dir = 'result_delta/result_%.3d.npz'%task_id
    
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

    # hyperparameters
    alpha_T = 1/np.std(Ts)
    alpha_X = 1

    GW_data = GW(alpha_T, alpha_X)
    L1_wavg = L1(wavg_mu, wavg_pi, wavg)
    L1_avg = L1(avg_mu, avg_pi, avg)
    
    # counterfactual of interest
    i0 = 0
    T0s_0 = Ts.copy(); T0s_0[:] = 0
    T0s_1 = Ts.copy(); T0s_1[:] = 1
    
    i0s = np.array([i0])
    T0s = np.array([T0s_0, T0s_1])
    
    # fit
    model_avg = KECENI.Model(
        LinearRegressionModel(avg_mu),
        LogisticIIDPropensityModel(avg_pi),
        IIDCovariateModel(bal=False),
        L1_avg.delta,
    )
    model_wavg = KECENI.Model(
        LinearRegressionModel(avg_mu),
        LogisticIIDPropensityModel(avg_pi),
        IIDCovariateModel(bal=False),
        L1_wavg.delta,
    )
    model_GW = KECENI.Model(
        LinearRegressionModel(avg_mu),
        LogisticIIDPropensityModel(avg_pi),
        IIDCovariateModel(bal=False),
        GW_data.delta,
    )

    # estimation
    fit_avg = model_avg.fit(data, tqdm=tqdm)
    fit_wavg = model_wavg.fit(
        data, tqdm=tqdm,
        mus=fit_avg.mus, pis=fit_avg.pis, ms=fit_avg.ms, vps=fit_avg.vps
    )
    fit_GW = model_GW.fit(
        data, tqdm=tqdm,
        mus=fit_avg.mus, pis=fit_avg.pis, ms=fit_avg.ms, vps=fit_avg.vps
    )
    
    ## G-computation
    result_G = fit_avg.G_estimate(i0s, T0s, n_X=n_X)
    YG_0, YG_1 = np.moveaxis(result_G, -1, 0)
    YG_d = YG_1 - YG_0
    
    ## cross-validation
    cv_avg = fit_avg.cv(
        tqdm=tqdm
    )
    cv_wavg = fit_wavg.cv(
        tqdm=tqdm
    )
    cv_GW = fit_GW.cv(
        tqdm=tqdm
    )

    xs_wavg, xhs_wavg = cv_wavg.xs_xhs(lamdas, kernel=exponential_kernel)
    xs_avg, xhs_avg = cv_avg.xs_xhs(lamdas, kernel=exponential_kernel)
    xs_GW, xhs_GW = cv_GW.xs_xhs(lamdas, kernel=exponential_kernel)

    id_avg = np.argmin(np.mean((xs_avg-xhs_avg)**2, -1))
    id_wavg = np.argmin(np.mean((xs_wavg-xhs_wavg)**2, -1))
    id_GW = np.argmin(np.mean((xs_GW-xhs_GW)**2, -1))
    
    ## KECENI
    result_avg = fit_avg.kernel_AIPW(
        i0s, T0s, tqdm=tqdm
    )
    YDR_0_avg, YDR_1_avg = np.moveaxis(result_avg.est(lamdas, exponential_kernel), -1, 0)
    YDR_d_avg = YDR_1_avg - YDR_0_avg

    result_wavg = fit_wavg.kernel_AIPW(
        i0s, T0s, tqdm=tqdm
    )
    YDR_0_wavg, YDR_1_wavg = np.moveaxis(result_wavg.est(lamdas, exponential_kernel), -1, 0)
    YDR_d_wavg = YDR_1_wavg - YDR_0_wavg

    result_GW = fit_GW.kernel_AIPW(
        i0s, T0s, tqdm=tqdm
    )
    YDR_0_GW, YDR_1_GW = np.moveaxis(result_GW.est(lamdas, exponential_kernel), -1, 0)
    YDR_d_GW = YDR_1_GW - YDR_0_GW
    
    # save results
    np.savez(save_dir, YG_0 = YG_0, YG_1=YG_1, YG_d=YG_d,
             xs_avg=xs_avg, xhs_avg=xhs_avg, id_avg=id_avg,
             xs_wavg=xs_wavg, xhs_wavg=xhs_wavg, id_wavg=id_wavg,
             xs_GW=xs_GW, xhs_GW=xhs_GW, id_GW=id_GW,
             YDR_0_avg=YDR_0_avg, YDR_1_avg=YDR_1_avg, YDR_d_avg=YDR_d_avg,
             YDR_0_wavg=YDR_0_wavg, YDR_1_wavg=YDR_1_wavg, YDR_d_wavg=YDR_d_wavg,
             YDR_0_GW=YDR_0_GW, YDR_1_GW=YDR_1_GW, YDR_d_GW=YDR_d_GW)


if __name__ == "__main__":
    # Get task ID from SLURM_ARRAY_TASK_ID
    if len(sys.argv) < 2:
        for i in range(n_sim):
            run_task(i)
    else:
        task_id = int(sys.argv[1])

        # Run the simulation for this task ID
        run_task(task_id)

