"""
Extracted the state propagation bits to individual functions
"""
import logging

import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as spl

from utils import block_diag

class NoHessianMethod(Exception):
    """An exception triggered when the forward model isn't able to provide an
    estimation of the Hessian"""
    def __init__(self, message):
        self.message = message

def band_selecta(band):
    if band == 0:
        return np.array([0, 1, 6, 2])
    else:
        return np.array([3, 4, 6, 5])


def hessian_correction_pixel(gp, x0, C_obs_inv, innovation, band, nparams):
    selecta = band_selecta(band)
    ddH = gp.hessian(np.atleast_2d(x0[selecta]))
    big_ddH = np.zeros((nparams, nparams))
    for i, ii in enumerate(selecta):
        for j, jj in enumerate(selecta):
            big_ddH[ii, jj] = ddH.squeeze()[i, j]
    big_hessian_corr = big_ddH*C_obs_inv*innovation
    return big_hessian_corr


def hessian_correction(gp, x0, R_mat, innovation, mask, state_mask, band,
                       nparams):
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
                                                        innov, band, nparams)
        little_hess.append(hessian_corr)
    hessian_corr = block_diag(little_hess)
    return hessian_corr


def tip_prior():
    """The JRC-TIP prior in a convenient function which is fun for the whole
    family. Note that the effective LAI is here defined in transformed space
    where TLAI = exp(-0.5*LAIe).

    Returns
    -------
    The mean prior vector, covariance and inverse covariance matrices."""
    # broadly TLAI 0->7 for 1sigma
    sigma = np.array([0.12, 0.7, 0.0959, 0.15, 1.5, 0.2, 0.5])
    x0 = np.array([0.17, 1.0, 0.1, 0.7, 2.0, 0.18, np.exp(-0.5*1.5)])
    # The individual covariance matrix
    little_p = np.diag(sigma**2).astype(np.float32)
    little_p[5, 2] = 0.8862*0.0959*0.2
    little_p[2, 5] = 0.8862*0.0959*0.2

    inv_p = np.linalg.inv(little_p)
    return x0, little_p, inv_p


def propagate_standard_kalman(x_analysis, P_analysis, P_analysis_inverse,
                              M_matrix, Q_matrix):
    """Standard Kalman filter state propagation using the state covariance
    matrix and a linear state transition model. This function returns `None`
    for the forecast inverse covariance matrix.

    Parameters
    -----------
    x_analysis : array
        The analysis state vector. This comes either from the assimilation or
        directly from a previoulsy propagated state.
    P_analysis : 2D sparse array
        The analysis covariance matrix (typically will be a sparse matrix).
    P_analysis_inverse : 2D sparse array
        The INVERSE analysis covariance matrix (typically a sparse matrix).
        As this is a Kalman update, you will typically pass `None` to it, as
        it is unused.
    M_matrix : 2D array
        The linear state propagation model.
    Q_matrix: 2D array (sparse)
        The state uncertainty inflation matrix that is added to the covariance
        matrix.

    Returns
    -------
    x_forecast (forecast state vector), P_forecast (forecast covariance matrix)
    and `None`"""

    x_forecast = M_matrix.dot(x_analysis)
    P_forecast = P_analysis + Q_matrix
    return x_forecast, P_forecast, None


def propagate_information_filter_SLOW(x_analysis, P_analysis, P_analysis_inverse,
                                 M_matrix, Q_matrix):
    """Information filter state propagation using the INVERSER state covariance
    matrix and a linear state transition model. This function returns `None`
    for the forecast covariance matrix (as this takes forever). This method is
    based on the approximation to the inverse of the KF covariance matrix.

    Parameters
    -----------
    x_analysis : array
        The analysis state vector. This comes either from the assimilation or
        directly from a previoulsy propagated state.
    P_analysis : 2D sparse array
        The analysis covariance matrix (typically will be a sparse matrix).
        As this is an information filter update, you will typically pass `None`
        to it, as it is unused.
    P_analysis_inverse : 2D sparse array
        The INVERSE analysis covariance matrix (typically a sparse matrix).
    M_matrix : 2D array
        The linear state propagation model.
    Q_matrix: 2D array (sparse)
        The state uncertainty inflation matrix that is added to the covariance
        matrix.

    Returns
    -------
    x_forecast (forecast state vector), `None` and P_forecast_inverse (forecast
    inverse covariance matrix)"""
    logging.info("Starting the propagation...")
    x_forecast = M_matrix.dot(x_analysis)
    n, n = P_analysis_inverse.shape
    S= P_analysis_inverse.dot(Q_matrix)
    A = (sp.eye(n) + S).tocsc()
    P_forecast_inverse = spl.spsolve(A, P_analysis_inverse)
    logging.info("DOne with propagation")

    return x_forecast, None, P_forecast_inverse

def propagate_information_filter_SLOW(x_analysis, P_analysis, P_analysis_inverse,
                                 M_matrix, Q_matrix):
    """Information filter state propagation using the INVERSER state covariance
    matrix and a linear state transition model. This function returns `None`
    for the forecast covariance matrix (as this takes forever). This method is
    based on calculating the actual matrix from the inverse of the inverse
    covariance, so it is **SLOW**. Mostly here for testing purposes.

    Parameters
    -----------
    x_analysis : array
        The analysis state vector. This comes either from the assimilation or
        directly from a previoulsy propagated state.
    P_analysis : 2D sparse array
        The analysis covariance matrix (typically will be a sparse matrix).
        As this is an information filter update, you will typically pass `None`
        to it, as it is unused.
    P_analysis_inverse : 2D sparse array
        The INVERSE analysis covariance matrix (typically a sparse matrix).
    M_matrix : 2D array
        The linear state propagation model.
    Q_matrix: 2D array (sparse)
        The state uncertainty inflation matrix that is added to the covariance
        matrix.

    Returns
    -------
    x_forecast (forecast state vector), `None` and P_forecast_inverse (forecast
    inverse covariance matrix)"""

    x_forecast = M_matrix.dot(x_analysis)
    # These is an approximation to the information filter equations
    # (see e.g. Terejanu's notes)
    M = P_analysis_inverse   # for convenience and to stay with
    #  Terejanu's notation
    # Main assumption here is that the "inflation" factor is
    # calculated using the main diagonal of M
    D = 1./(1. + M.diagonal()*Q_matrix.diagonal())
    M = sp.dia_matrix((M.diagonal(), 0), shape=M.shape)
    P_forecast_inverse = M.dot(sp.dia_matrix((D, 0),
                                             shape=M.shape))
    return x_forecast, None, P_forecast_inverse


def propagate_information_filter_LAI(x_analysis, P_analysis,
                                     P_analysis_inverse,
                                     M_matrix, Q_matrix):


    x_forecast = M_matrix.dot(x_analysis)
    x_prior, c_prior, c_inv_prior = tip_prior()
    n_pixels = len(x_analysis)/7
    x0 = np.array([x_prior for i in xrange(n_pixels)]).flatten()
    x0[6::7] = x_forecast[6::7] # Update LAI
    print "LAI:", -2*np.log(x_forecast[6::7])
    lai_post_cov = P_analysis_inverse.diagonal()
    c_inv_prior_mat = []
    for n in xrange(n_pixels):
        c_inv_prior[6,6] =  lai_post_cov[n]
        c_inv_prior_mat.append(c_inv_prior)

    P_forecast_inverse=block_diag(c_inv_prior_mat, dtype=np.float32)

    return x0, None, P_forecast_inverse

def no_propagation(x_analysis, P_analysis,
                                     P_analysis_inverse,
                                     M_matrix, Q_matrix):
    """No propagation. In this case, we return the original prior. As the
    information filter behaviour is the standard behaviour in KaFKA, we
    only return the inverse covariance matrix. **NOTE** the input parameters
    are there to comply with the API, but are **UNUSED**.

    Parameters
    -----------
    x_analysis : array
        The analysis state vector. This comes either from the assimilation or
        directly from a previoulsy propagated state.
    P_analysis : 2D sparse array
        The analysis covariance matrix (typically will be a sparse matrix).
        As this is an information filter update, you will typically pass `None`
        to it, as it is unused.
    P_analysis_inverse : 2D sparse array
        The INVERSE analysis covariance matrix (typically a sparse matrix).
    M_matrix : 2D array
        The linear state propagation model.
    Q_matrix: 2D array (sparse)
        The state uncertainty inflation matrix that is added to the covariance
        matrix.

    Returns
    -------
    x_forecast (forecast state vector), `None` and P_forecast_inverse (forecast
    inverse covariance matrix)"""

    x_prior, c_prior, c_inv_prior = tip_prior()
    n_pixels = len(x_analysis)/7
    x_forecast = np.array([x_prior for i in xrange(n_pixels)]).flatten()
    c_inv_prior_mat = [c_inv_prior for n in xrange(n_pixels)]
    P_forecast_inverse=block_diag(c_inv_prior_mat, dtype=np.float32)

    return x_forecast, None, P_forecast_inverse
