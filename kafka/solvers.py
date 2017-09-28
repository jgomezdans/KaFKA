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

    LOG.info("Diagonal covariance solver")
    the_mask = np.array([mask.ravel() for i in xrange(n_params)]).ravel() 

    S = (H_matrix.dot(P_forecast.dot(H_matrix.T))) + R_mat
    Sd = np.zeros(x_forecast.shape[0]/n_params)
    Sd[mask.ravel()] = 1./(S.diagonal()[mask.ravel()])
    Sinv = sp.eye(Sd.shape[0])
    Sinv.setdiag(Sd)
    
    kalman_gain = (P_forecast.dot(H_matrix.T)).dot(Sinv)
  
    innovations = (observations.ravel() - H_matrix.dot(x_forecast))
    innovations[~mask.ravel()] = 0.
    x_analysis = x_forecast + kalman_gain*innovations
    P_analysis = (sp.eye(x_analysis.shape[0]) -
                  kalman_gain.dot(H_matrix)).dot(P_forecast)
    P_analysis_prime = None
    LOG.info("Solved!")
    return x_analysis, P_analysis, None, innovations[~mask.ravel()]


def kalman_divide_conquer( observations, mask, H_matrix, n_params,
            x_forecast, P_forecast, the_metadata, approx_diagonal=True):
    """This function solves the problem "one pixel at a time" kind of strategy
    """

def variational_kalman( observations, mask, uncertainty, H_matrix, n_params,
            x_forecast, P_forecast, P_forecast_inv, the_metadata, approx_diagonal=True):
    """We can just use """

    if len(H_matrix) == 2:
        non_linear = True
        H0, H_matrix = H_matrix
    else:
        H0 = 0.
        non_linear = False
        
    LOG.info("Squeezing prior covariance...")
    R_mat = uncertainty
    maska = np.concatenate([mask.ravel() for i in xrange(n_params)]) 
    #Pinv = 1./np.array(P_forecast.diagonal()).squeeze()[maska]
    #P_forecast_inv = sp.eye(Pinv.shape[0])
    #P_forecast_inv.setdiag(Pinv)
    LOG.info("Creating linear problem")
    
    y = np.zeros_like(observations)
    y[mask] = observations[mask]
    y = y.ravel()
    #y = observations.ravel()#[mask.ravel()]
    #y[~mask.ravel()] = 0.
    if non_linear:
        y = y + H_matrix.dot(x_forecast) - H0
    
        
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
    
