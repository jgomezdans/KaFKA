#!/usr/bin/env python
"""Some solvers"""

# KaFKA A fast Kalman filter implementation for raster based datasets.
# Copyright (c) 2017 J Gomez-Dans. All rights reserved.
#
# This file is part of KaFKA.
#
# KaFKA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KaFKA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KaFKA.  If not, see <http://www.gnu.org/licenses/>.

from collections import namedtuple
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

#from utils import  matrix_squeeze, spsolve2, reconstruct_array

# Set up logging
import logging
LOG = logging.getLogger(__name__+".solvers")


__author__ = "J Gomez-Dans"
__copyright__ = "Copyright 2017 J Gomez-Dans"
__version__ = "1.0 (09.03.2017)"
__license__ = "GPLv3"
__email__ = "j.gomez-dans@ucl.ac.uk"


####def variational_kalman( observations, mask, state_mask, uncertainty, H_matrix, n_params,
            ####x_forecast, P_forecast, P_forecast_inv, the_metadata, approx_diagonal=True):
    ####"""We can just use """
    ####if len(H_matrix) == 2:
        ####non_linear = True
        ####H0, H_matrix_ = H_matrix
    ####else:
        ####H0 = 0.
        ####non_linear = False
    ####R_mat = sp.diags(uncertainty.diagonal()[state_mask.flatten()])
    ####LOG.info("Creating linear problem")
    ####y = observations[state_mask]
    ####y = np.where(mask[state_mask], y, 0.)
    ####y_orig = y*1.
    ####if non_linear:
        ####y = y + H_matrix_.dot(x_forecast) - H0
    
        
    #####Aa = matrix_squeeze (P_forecast_inv, mask=maska.ravel())
    ####A = H_matrix_.T.dot(R_mat).dot(H_matrix_) + P_forecast_inv
    ####b = H_matrix_.T.dot(R_mat).dot(y) + P_forecast_inv.dot (x_forecast)
    ####b = b.astype(np.float32)
    ####A = A.astype(np.float32)
    ##### Here we can either do a spLU of A, and solve, or we can have a first go
    ##### by assuming P_forecast_inv is diagonal, and use the inverse of A_approx as
    ##### a preconditioner
    ####LOG.info("Solving")
    ####AI = sp.linalg.splu (A)
    ####x_analysis = AI.solve (b)
    ##### So retval is the solution vector and A is the Hessian 
    ##### (->inv(A) is posterior cov)
    ####fwd_modelled = H_matrix_.dot(x_analysis-x_forecast) + H0
    ####innovations = y_orig - fwd_modelled
    
    #####x_analysis = reconstruct_array ( x_analysis_prime, x_forecast,
    #####                                    mask.ravel(), n_params=n_params)
    
    ####return x_analysis, None, A, innovations, fwd_modelled
    

####def sort_band_data(H_matrix, observations, uncertainty, mask, 
                   ####x0, x_forecast, state_mask):
    ####if len(H_matrix) == 2:
        ####non_linear = True
        ####H0, H_matrix_ = H_matrix
    ####else:
        ####H0 = 0.
        ####H_matrix_ = H_matrix
        ####non_linear = False
    ####R = uncertainty.diagonal()[state_mask.flatten()]
    ####y = observations[state_mask]
    ####y = np.where(mask[state_mask], y, 0.)
    ####y_orig = y*1.
    ####if non_linear:
        ####y = y + H_matrix_.dot(x0) - H0
    ####return H_matrix_, H0, R, y, y_orig
        


####def variational_kalman_multiband(observations_b, mask_b, state_mask, 
                                 ####uncertainty_b, H_matrix_b, n_params,
                                 ####x0, x_forecast, P_forecast, P_forecast_inv,
                                 ####the_metadata_b, approx_diagonal=True,
                                 ####upper_bound=None, lower_bound=None):
    ####"""
    
    ####Parameters
    ####-----------
    
    ####observations_b: list
        ####A list of the observations (each band a 2D array).
    ####mask_b: list
        ####A list of per band masks (2D arrays)
    ####state_mask: array
        ####A 2D boolean array with the state mask
    ####uncertainty_b: list
        ####A list of inverse covariance matrices for each pixel
    ####H_matrix_b: list
        ####A list of (i) linear operators or (ii) Jacobian and value of
        ####H model at the linearisation point, x0.
    ####n_params: int
        ####Number of parameters
    ####x0: array
        ####Vector with the linearsation point. Should be n_params*state_mask.sum()
    ####x_forecast: array
        ####This is in effect the prior mean. On the first iteratin, x_forecast=x0, but
        ####then x0 will change
    ####P_forecast: sparse matrix 
        ####the prior covariance matrix. Often not used
    ####P_forecast_inv: sparse matrix
        ####The prior **inverse** covariance matrix. This is the one we use all the time
    ####the_metadata_b: list
        ####A list of metadata. This is dependent on the observation type
    ####approx_diagonal: boolean
        ####Wehther to approximation something by its diagonal, which isn't used here.
    ####upper_bound: dict
        ####A dictionary with the upper parameter bounds. Just clips the solution, so not
        ####sure this ought to be used or not
    ####lower_bound: dict
        ####A dictionary with the lower parameter bounds. Just clips the solution, so not
        ####sure this ought to be used or not
        
    ####"""
    ####n_bands = len(observations_b)
    
    ####y = []
    ####y_orig = []
    ####H_matrix = []
    ####H0 = []
    ####R_mat = []
    ####for i in range(n_bands):
        ####a, b, c, d, e = sort_band_data(H_matrix_b[i], observations_b[i],
                                       ####uncertainty_b[i], mask_b[i], x0,
                                       ####x_forecast, state_mask)
        ####H_matrix.append(a)
        ####H0.append(b)
        ####R_mat.append(c)
        ####y.append(d)
        ####y_orig.append(e)
    ####H_matrix_ = sp.vstack(H_matrix)
    ####H0 = np.hstack(H0)
    ####R_mat = sp.diags(np.hstack(R_mat))
    ####y = np.hstack(y)
    ####y_orig = np.hstack(y_orig)
    
    #####Aa = matrix_squeeze (P_forecast_inv, mask=maska.ravel())
    ####A = H_matrix_.T.dot(R_mat).dot(H_matrix_) + P_forecast_inv
    #####A = H_matrix_.T @ (R_mat @ H_matrix_) + P_forecast_inv
    ####b = H_matrix_.T.dot(R_mat).dot(y) + P_forecast_inv.dot (x_forecast)
    ##### b = H_matrix_.T @ (R_mat @ y) + P_forecast_inv @ x_forecast
    ####b = b.astype(np.float32)
    ####A = A.astype(np.float32)
    ##### Here we can either do a spLU of A, and solve, or we can have a first go
    ##### by assuming P_forecast_inv is diagonal, and use the inverse of A_approx as
    ##### a preconditioner
    ####LOG.info("Solving")
    ####AI = sp.linalg.splu (A)
    ####x_analysis = AI.solve (b)
    ####n_elems = n_params*state_mask.sum()
    ####if upper_bound is None:
        ####hi = None
    ####else:
        ####hi = np.zeros_like(x_analysis)
        ####for j, (k, v) in enumerate(upper_bound.items()):
            ####hi[j::n_params] = v

        
    ####if lower_bound is  None:
        ####lo = None
    ####else:
        ####lo = np.zeros_like(x_analysis)
        ####for j, (k, v) in enumerate(lower_bound.items()):
            ####lo[j::n_params] = v
        

    ####x_analysis = np.clip(x_analysis, lo, hi)
    ##### So retval is the solution vector and A is the Hessian 
    ##### (->inv(A) is posterior cov)
    ####fwd_modelled = H_matrix_.dot(x_analysis-x_forecast) + H0
    ####innovations = y_orig - fwd_modelled
    
    ####return x_analysis, None, A, innovations, fwd_modelled


def variational_kalman( observations, mask, state_mask, uncertainty, H_matrix, n_params,
            x_forecast, P_forecast, P_forecast_inv, the_metadata, approx_diagonal=True,
            upper_bound=None, lower_bound=None):
    """We can just use """
    if len(H_matrix) == 2:
        non_linear = True
        H0, H_matrix_ = H_matrix
    else:
        H0 = 0.
        non_linear = False
    R_mat = sp.diags(uncertainty.diagonal()[state_mask.flatten()])
    LOG.info("Creating linear problem")
    y = observations[state_mask]
    y = np.where(mask[state_mask], y, 0.)
    y_orig = y*1.
    if non_linear:
        y = y + H_matrix_.dot(x_forecast) - H0
    
        
    #Aa = matrix_squeeze (P_forecast_inv, mask=maska.ravel())
    A = H_matrix_.T.dot(R_mat).dot(H_matrix_) + P_forecast_inv
    b = H_matrix_.T.dot(R_mat).dot(y) + P_forecast_inv.dot (x_forecast)
    b = b.astype(np.float32)
    A = A.astype(np.float32)
    # Here we can either do a spLU of A, and solve, or we can have a first go
    # by assuming P_forecast_inv is diagonal, and use the inverse of A_approx as
    # a preconditioner
    LOG.info("Solving")
    AI = sp.linalg.splu (A)
    x_analysis = AI.solve (b)
    # So retval is the solution vector and A is the Hessian 
    # (->inv(A) is posterior cov)
    fwd_modelled = H_matrix_.dot(x_analysis-x_forecast) + H0
    innovations = y_orig - fwd_modelled
    
    #x_analysis = reconstruct_array ( x_analysis_prime, x_forecast,
    #                                    mask.ravel(), n_params=n_params)
    
    return x_analysis, None, A, innovations, fwd_modelled
    

def sort_band_data(H_matrix, observations, uncertainty, mask, 
                   x0, x_forecast, state_mask):
    if len(H_matrix) == 2:
        non_linear = True
        H0, H_matrix_ = H_matrix
    else:
        H0 = 0.
        H_matrix_ = H_matrix
        non_linear = False
    R = uncertainty.diagonal()[state_mask.flatten()]
    y = observations[state_mask]
    y = np.where(mask[state_mask], y, 0.)
    y_orig = y*1.
    if non_linear:
        y = y + H_matrix_.dot(x0) - H0
    return H_matrix_, H0, R, y, y_orig
        


def variational_kalman_multiband( observations_b, mask_b, state_mask, uncertainty_b, H_matrix_b, n_params,
            x0, x_forecast, P_forecast, P_forecast_inv, the_metadata_b, approx_diagonal=True,
            upper_bound=None, lower_bound=None):
    """We can just use """
    n_bands = len(observations_b)
    
    y = []
    y_orig = []
    H_matrix = []
    H0 = []
    R_mat = []
    for i in range(n_bands):
        a, b, c, d, e = sort_band_data(H_matrix_b[i], observations_b[i], 
                                       uncertainty_b[i], mask_b[i], x0, x_forecast, state_mask)
        H_matrix.append(a)
        H0.append(b)
        R_mat.append(c)
        y.append(d)
        y_orig.append(e)
    H_matrix_ = sp.vstack(H_matrix)
    H0 = np.hstack(H0)
    R_mat = sp.diags(np.hstack(R_mat))
    y = np.hstack(y)
    y_orig = np.hstack(y_orig)
    
    #Aa = matrix_squeeze (P_forecast_inv, mask=maska.ravel())
    A = H_matrix_.T.dot(R_mat).dot(H_matrix_) + P_forecast_inv
    b = H_matrix_.T.dot(R_mat).dot(y) + P_forecast_inv.dot (x_forecast)
    b = b.astype(np.float32)
    A = A.astype(np.float32)
    # Here we can either do a spLU of A, and solve, or we can have a first go
    # by assuming P_forecast_inv is diagonal, and use the inverse of A_approx as
    # a preconditioner
    LOG.info("Solving")
    AI = sp.linalg.splu (A)
    x_analysis = AI.solve (b)
    # So retval is the solution vector and A is the Hessian 
    # (->inv(A) is posterior cov)
    fwd_modelled = H_matrix_.dot(x_analysis-x_forecast) + H0
    innovations = y_orig - fwd_modelled
    """ For now I am going to return innovations as y_orig - H0 as
    That is what is needed by the Hessian correction. Need to discuss with Jose 
    What the intention for innovations is and then we can find the best solution"""
    innovations = y_orig - H0
    #x_analysis = reconstruct_array ( x_analysis_prime, x_forecast,
    #                                    mask.ravel(), n_params=n_params)
    return x_analysis, None, A, innovations, fwd_modelled
    
