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
LOG = logging.getLogger(__name__)


__author__ = "J Gomez-Dans"
__copyright__ = "Copyright 2017 J Gomez-Dans"
__version__ = "1.0 (09.03.2017)"
__license__ = "GPLv3"
__email__ = "j.gomez-dans@ucl.ac.uk"


def linear_diagonal_solver ( observations, mask, H_matrix, n_params,
            x_forecast, P_forecast, R_mat, the_metadata, approx_diagonal=True):
    LOG.info("Squeezing prior covariance...")
                                  #n_params=self.n_params)
    P_forecast_prime = np.array(P_forecast.diagonal()).squeeze()[mask.ravel()]
                

    # At this stage, we have a forecast (prior), the observations
    # and the observation operator, so we proceed with the
    # assimilation
    if approx_diagonal:
        # We approximate the inverse matrix by a division assuming
        # P_forecast is diagonal
        LOG.info("Diagonal approximation")
        R_mat_prime = np.array(R_mat.diagonal()).squeeze()
        S = H_matrix.dot(H_matrix.T)*P_forecast_prime + R_mat_prime
        nn1 = R_mat_prime.shape[0]
        LOG.info("About to calculate approx KGain")
        kalman_gain = P_forecast_prime*np.array(H_matrix.diagonal()).squeeze()
        LOG.info("About to calculate approx KGain")
        kalman_gain = kalman_gain/S
    LOG.info("Squeeze x_forecast")
    x_forecast_prime = matrix_squeeze(x_forecast, mask=mask.ravel(),
                                                          n_params=n_params)
    LOG.info("Calculating innovations")
    innovations_prime = (observations.ravel()[mask.ravel()] -
                                             H_matrix.dot(x_forecast_prime))
    LOG.info("Calculating analysis state")
    x_analysis_prime = x_forecast_prime + \
                                           kalman_gain*innovations_prime
    LOG.info("Calculating analysis covariance")
    P_analysis_prime = (np.ones_like(P_forecast_prime) -
                        kalman_gain*H_matrix)*P_forecast_prime
    tmp_matrix = sp.eye(nn1)
    tmp_matrix.setdiag(P_analysis_prime)
    P_analysis_prime = tmp_matrix

    # Now move
    LOG.info("Inflating analysis state")
    x_analysis = reconstruct_array ( x_analysis_prime, x_forecast,
                                        mask.ravel(), n_params=n_params)
    LOG.info("Analsysis smalld diagonal, useful as preconditioner")
    small_diagonal = np.array(P_analysis_prime.diagonal()).squeeze()
    big_diagonal = np.array(P_forecast.diagonal()).squeeze()
    LOG.info("Inflate analysis covariance")
    P_analysis_diag = reconstruct_array(small_diagonal, big_diagonal,
                                    mask, n_params=n_params)
    P_analysis = sp.dia_matrix ( (P_analysis_diag, 0),
                                    shape=P_forecast.shape)
    return x_analysis, P_analysis, innovations_prime


def kalman_divide_conquer( observations, H_matrix, n_params,
            x_forecast, P_forecast, the_metadata, approx_diagonal=True):
    """This function solves the problem "one pixel at a time" kind of strategy
    """

def variational_kalman( observations, H_matrix, n_params,
            x_forecast, P_forecast, P_forecast_inv, the_metadata, approx_diagonal=True):
    """We can just use """
    LOG.info("Squeezing prior covariance...")
    mask = the_metadata.mask
    maska = np.concatenate([mask.ravel() for i in xrange(n_params)]) 
    #Pinv = 1./np.array(P_forecast.diagonal()).squeeze()[maska]
    #P_forecast_inv = sp.eye(Pinv.shape[0])
    #P_forecast_inv.setdiag(Pinv)
    LOG.info("Creating linear problem")
    R_mat = the_metadata.uncertainty
    y = observations.ravel()#[mask.ravel()]
    y[~mask.ravel()] = 0.
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
