"""
Extracted the state propagation bits to individual functions
"""
import numpy as np

import scipy.sparse as sp

from utils import block_diag


def tip_prior():
    sigma = np.array([0.12, 0.7, 0.0959, 0.15, 1.5, 0.2, 0.5]) # broadly TLAI 0->7 for 1sigma
    x0 = np.array([0.17, 1.0, 0.1, 0.7, 2.0, 0.18, np.exp(-0.5*1.5)])
    # The individual covariance matrix
    little_p = np.diag ( sigma**2).astype(np.float32)
    little_p[5,2] = 0.8862*0.0959*0.2
    little_p[2,5] = 0.8862*0.0959*0.2
    
    inv_p = np.linalg.inv(little_p)
    return x0, little_p, inv_p


def propagate_standard_kalman(x_analysis, P_analysis, P_analysis_inverse,
                              M_matrix, Q_matrix):
    x_forecast = M_matrix.dot(x_analysis)
    P_forecast = P_analysis + Q_matrix
    return x_forecast, P_forecast, None


def propagate_information_filter(x_analysis, P_analysis, P_analysis_inverse,
                                 M_matrix, Q_matrix):
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
    
    x_prior, c_prior, c_inv_prior = tip_prior()
    n_pixels = len(x_analysis)/7
    x_forecast = np.array([x_prior for i in xrange(n_pixels)]).flatten()
    c_inv_prior_mat = [c_inv_prior for n in xrange(n_pixels)]
    P_forecast_inverse=block_diag(c_inv_prior_mat, dtype=np.float32)

    return x_forecast, None, P_forecast_inverse
