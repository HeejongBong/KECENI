import numpy as np
from .Fit import Fit
from .KernelEstimate import KernelEstimate, CrossValidationEstimate

def concat_Fits(list_fit):
    # def __init__(self, model, data, n_X=100, js=None, 
    #              mus=None, pis=None, ms=None, vps=None,
    #              n_process=None, tqdm=None, level_tqdm=0):
    
    return Fit(
        list_fit[0].model, list_fit[0].data,
        np.min([fit_i.n_X for fit_i in list_fit]),
        np.concatenate([fit_i.js for fit_i in list_fit]),
        np.concatenate([fit_i.mus for fit_i in list_fit]),
        np.concatenate([fit_i.pis for fit_i in list_fit]),
        np.concatenate([fit_i.ms for fit_i in list_fit]),
        np.concatenate([fit_i.vps for fit_i in list_fit]),
    )


def concat_KEs(list_KE):
    # def __init__(self, fit, i0s, T0s, G0, Ds):
    return KernelEstimate(
        concat_Fits([KE_i.fit for KE_i in list_KE]),
        list_KE[0].i0s, list_KE[0].T0s, list_KE[0].G0,
        np.concatenate([KE_i.Ds for KE_i in list_KE]),
    )

def concat_CVs(list_CV):
    # def __init__(self, fit, i0s, T0s, G0, Ds):
    return CrossValidationEstimate(
        concat_Fits([CV_i.fit for CV_i in list_CV]),
        list_CV[0].i0s, list_CV[0].T0s, list_CV[0].G0,
        np.concatenate([CV_i.Ds for CV_i in list_CV]),
    )

def concat_phis(list_ws, list_phis, list_Hs=None):
    if list_Hs is None:
        return np.concatenate([
            phis_i * ws_i / np.sum(list_ws, 0)
            for phis_i, ws_i in zip(list_phis, list_ws)
        ])
    else:
        phis_eif = concat_phis(list_ws, list_phis)
        return (
            phis_eif + np.sum(
                np.moveaxis(list_Hs, 0, -np.array(list_ws).ndim)
                * np.array(list_ws), -np.array(list_ws).ndim
            ) / np.sum(list_ws, 0)
        )