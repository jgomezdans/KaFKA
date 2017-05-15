#!/usr/bin/env python
"""A fast Kalman filter implementation designed with raster data in mind. This
implementation basically performs a very fast update of the filter."""

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

__author__ = "J Gomez-Dans"
__copyright__ = "Copyright 2017 J Gomez-Dans"
__version__ = "1.0 (09.03.2017)"
__license__ = "GPLv3"
__email__ = "j.gomez-dans@ucl.ac.uk"

import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = "1"
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from linear_kf import LinearKalman
from solvers import variational_kalman, kalman_divide_conquer
import gp_emulator
import gdal
# metadata is now different as it has angles innit
Metadata = namedtuple('Metadata', 'mask uncertainty band')
MCD43_observations = namedtuple('MCD43_observations',
                        'doys mcd43a1 mcd43a2')


# Set up logging
import logging
LOG = logging.getLogger(__name__+".nonlinear_ekf")

class NonLinearKalman (LinearKalman):
    """A class that extends to non linear models"""

    def __init__(self, emulator, observations, observation_times,
                 observation_metadata, output_array, output_unc,
                 bands_per_observation=2, diagnostics=True, n_params=7):

        LinearKalman.__init__(self, observations, observation_times,
                              observation_metadata, output_array, output_unc,
                              bands_per_observation=bands_per_observation,
                              diagnostics=diagnostics,
                              n_params=n_params)
        self.emulator = emulator
        
    def _dump_output(self, ii, timestep, x_analysis, P_analysis, 
                               P_analysis_inverse):
        pass

    def create_observation_operator(self, metadata, x_forecast, band):
        """Using an emulator of the nonlinear model around `x_forecast`.
        This case is quite special, as I'm focusing on a BHR SAIL 
        version (or the JRC TIP), which have spectral parameters 
        (e.g. leaf single scattering albedo in two bands, etc.). This
        is achieved by using the `state_mapper` to select which bits
        of the state vector (and model Jacobian) are used."""
        LOG.info("Creating the ObsOp for band %d" % band)
        n_times = x_forecast.shape[0]/self.n_params
        good_obs = metadata.mask.sum()
        
        
        H_matrix = sp.lil_matrix ( (n_times, self.n_params*n_times),
                                 dtype=np.float32)

        H0 = np.zeros(n_times, dtype=np.float32)

        # So the model has spectral components. 
        if band == 0:
            # ssa, asym, LAI, rsoil
            state_mapper = np.array([0,1,6,2])
        elif band == 1:
            # ssa, asym, LAI, rsoil
            state_mapper = np.array([3,4,6,5])
        
        x0 = np.zeros((n_times, 4))
        for i in xrange(n_times):
            if metadata.mask.ravel()[i]:
                x0[i, :] = x_forecast[state_mapper + self.n_params*i]
        LOG.info("Running emulators") 
        H0_, dH = self.emulator.predict(x0[metadata.mask.ravel()], do_unc=False)
        n = 0
        LOG.info("Storing emulators in H matrix")
        for i in xrange(n_times):
            if metadata.mask.ravel()[i]:
                H_matrix[i, state_mapper + self.n_params*i] = dH[n]
                H0[i] = H0_[n]
                n += 1
        LOG.info("\tDone!")
        return (H0, H_matrix.tocsr())

    def solver(self, observations, mask, H_matrix, x_forecast, P_forecast,
                P_forecast_inv, R_mat, the_metadata):
        x_analysis, P_analysis, P_analysis_inv, innovations_prime = \
            variational_kalman (
            observations, H_matrix, self.n_params, x_forecast,
            P_forecast, P_forecast_inv, the_metadata)
        
        return x_analysis, P_analysis, P_analysis_inv, innovations_prime
