"""
Extracted the state propagation bits to individual functions
"""
import logging

import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as spl

from .utils import block_diag

class NoHessianMethod(Exception):
    """An exception triggered when the forward model isn't able to provide an
    estimation of the Hessian"""
    def __init__(self, message):
        self.message = message


def hessian_correction_pixel(gp, x0, C_obs_inv, innovation, band, 
                             nparams, band_mapper):
    selecta = band_mapper[band]

    ddH = gp.hessian(np.atleast_2d(x0[selecta]))
    big_ddH = np.zeros((nparams, nparams))
    for i, ii in enumerate(selecta):
        for j, jj in enumerate(selecta):
            big_ddH[ii, jj] = ddH.squeeze()[i, j]
    big_hessian_corr = big_ddH*C_obs_inv*innovation
    return big_hessian_corr


def hessian_correction(gp, x0, R_mat, innovation, mask, state_mask, band,
                       nparams, band_mapper):
    """Calculates higher order Hessian correction for the likelihood term.
    Needs the GP, the Observational uncertainty, the mask...."""
    if not hasattr(gp, "hessian"):
        # The observation operator does not provide a Hessian method. We just
        # return 0, meaning no Hessian correction.
        return 0.
    C_obs_inv = R_mat.diagonal()[state_mask.flatten()]
    mask = mask[state_mask].flatten()
    little_hess = []
    for i, (innov, C, m) in enumerate(zip(innovation, C_obs_inv, mask)):
        if not m:
            # Pixel is masked
            hessian_corr = np.zeros((nparams, nparams))
        else:
            # Get state for current pixel
            x0_pixel = x0.squeeze()[(nparams*i):(nparams*(i + 1))]
            # Calculate the Hessian correction for this pixel
            hessian_corr = m * hessian_correction_pixel(gp, x0_pixel, C,
                                                        innov, band, nparams,
                                                        band_mapper)
        little_hess.append(hessian_corr)
    hessian_corr = block_diag(little_hess)
    return hessian_corr


def hessian_correction_multiband(gp, x0, R_mats, innovations, masks, state_mask, 
                                 n_bands, nparams, band_mapper):
    """ Non linear correction for the Hessian of the cost function. This handles
    multiple bands. """
    little_hess_cor = []
    for R, innovation, mask, band in zip(R_mats, innovations, masks, 
                                         range(n_bands)):
        little_hess_cor.append(hessian_correction(gp, x0, R, innovation, 
                                                  mask, state_mask, band,
                                                  nparams, band_mapper))
    hessian_corr = sum(little_hess_cor) #block_diag(little_hess_cor)
    return hessian_corr

