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


def variational_kalman( observations, mask, state_mask, uncertainty, H_matrix, n_params,
            x_forecast, P_forecast, P_forecast_inv, the_metadata, approx_diagonal=True):
    """We can just use """
    if len(H_matrix) == 2:
        non_linear = True
        H0, H_matrix = H_matrix
    else:
        H0 = 0.
        non_linear = False
    R_mat = sp.diags(uncertainty.diagonal()[state_mask.flatten()])
    LOG.info("Creating linear problem")
    y = observations[state_mask]
    y = np.where(mask[state_mask], y, 0.)
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
    fwd_modelled = H_matrix.dot(x_analysis)
    innovations = y - fwd_modelled
    
    LOG.info("Inflating analysis state")
    #x_analysis = reconstruct_array ( x_analysis_prime, x_forecast,
    #                                    mask.ravel(), n_params=n_params)
    
    return x_analysis, None, A, innovations, fwd_modelled
    
