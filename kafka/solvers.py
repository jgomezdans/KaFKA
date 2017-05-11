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

from utils import  matrix_squeeze, spsolve2, reconstruct_array

# Set up logging
import logging
LOG = logging.getLogger(__name__+".solvers")


__author__ = "J Gomez-Dans"
__copyright__ = "Copyright 2017 J Gomez-Dans"
__version__ = "1.0 (09.03.2017)"
__license__ = "GPLv3"
__email__ = "j.gomez-dans@ucl.ac.uk"


def linear_diagonal_solver ( observations, mask, H_matrix, n_params,
            x_forecast, P_forecast, R_mat, the_metadata, approx_diagonal=True):


    LOG.info("Squeezing prior covariance...")                                  
    the_mask = np.array([mask.ravel() for i in xrange(n_params)]).ravel() 
    # Only diagonal considered
    P_forecast_prime = np.array(P_forecast.diagonal()).squeeze()[the_mask]
                
    
    
    #R_mat_prime = np.array(R_mat.diagonal()).squeeze()
    A = sp.eye(P_forecast_prime.shape[0])
    A.setdiag(P_forecast_prime)
    S = (H_matrix.dot(A.dot(H_matrix.T))) + R_mat
    Sinv = sp.eye(S.shape[0])
    Sinv.setdiag(1./S.diagonal())
    nn1 = R_mat.shape[0]
    kalman_gain = A.dot(H_matrix.T)
    kalman_gain = kalman_gain.dot(Sinv)
    
    x_forecast_prime = matrix_squeeze(x_forecast, mask=mask.ravel(),
                                                          n_params=n_params)
    innovations_prime = (observations.ravel()[mask.ravel()] -
                                             H_matrix.dot(x_forecast_prime))
    x_analysis_prime = x_forecast_prime + \
                                           kalman_gain*innovations_prime
    P_analysis_prime = (sp.eye(n_params*nn1, n_params*nn1, 
                               dtype=np.float32) - (kalman_gain.dot(
                                   H_matrix)).dot(A))
    tmp_matrix = sp.eye(nn1)
    tmp_matrix.setdiag(P_analysis_prime.diagonal())
    P_analysis_prime = None
    P_analysis_prime = tmp_matrix
    
    # Now move
    
    x_analysis = reconstruct_array ( x_analysis_prime, x_forecast,
                                        mask.ravel(), n_params=n_params)
    small_diagonal = np.array(P_analysis_prime.diagonal()).squeeze()
    big_diagonal = np.array(P_forecast.diagonal()).squeeze()
    LOG.info("Inflate analysis covariance")
    P_analysis_diag = reconstruct_array(small_diagonal, big_diagonal,
                                    mask, n_params=n_params)
    P_analysis = sp.dia_matrix ( (P_analysis_diag, 0),
                                    shape=P_forecast.shape)
    return x_analysis, P_analysis, None, innovations_prime


def kalman_divide_conquer( observations, H_matrix, n_params,
            x_forecast, P_forecast, the_metadata, approx_diagonal=True):
    """This function solves the problem "one pixel at a time" kind of strategy
    """

def variational_kalman( observations, H_matrix, n_params,
            x_forecast, P_forecast, P_forecast_inv, the_metadata, approx_diagonal=True):
    """We can just use """
    
    H0, H_matrix = H_matrix
    LOG.info("Squeezing prior covariance...")
    mask = the_metadata.mask
    maska = np.concatenate([mask.ravel() for i in xrange(n_params)]) 
    #Pinv = 1./np.array(P_forecast.diagonal()).squeeze()[maska]
    #P_forecast_inv = sp.eye(Pinv.shape[0])
    #P_forecast_inv.setdiag(Pinv)
    LOG.info("Creating linear problem")
    R_mat = the_metadata.uncertainty
    y = np.zeros_like(observations)
    y[mask] = observations[mask]
    y = y.ravel()
    #y = observations.ravel()#[mask.ravel()]
    #y[~mask.ravel()] = 0.
    y = y - H0 + H_matrix.dot(x_forecast)
    #Aa = matrix_squeeze (P_forecast_inv, mask=maska.ravel())
    A = H_matrix.T.dot(R_mat).dot(H_matrix) + P_forecast_inv
    b = H_matrix.T.dot(R_mat).dot(y) + P_forecast_inv.dot (x_forecast)
    # Here we can either do a spLU of A, and solve, or we can have a first go
    # by assuming P_forecast_inv is diagonal, and use the inverse of A_approx as
    # a preconditioner
    LOG.info("Solving")
    AI = sp.linalg.splu ( A )
    x_analysis = AI.solve (b)
    # So retval is the solution vector and A is the Hessian 
    # (->inv(A) is posterior cov)
    innovations = y - H_matrix.dot(x_analysis)
    LOG.info("Inflating analysis state")
    #x_analysis = reconstruct_array ( x_analysis_prime, x_forecast,
    #                                    mask.ravel(), n_params=n_params)
    
    return x_analysis, None, A, innovations
