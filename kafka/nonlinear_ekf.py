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


from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from linear_kf import LinearKalman
import gp_emulator
import gdal
# metadata is now different as it has angles innit
Metadata = namedtuple('Metadata', 'mask uncertainty')
MCD43_observations = namedtuple('MCD43_observations',
                        'doys mcd43a1 mcd43a2')



class NonLinearKalman (LinearKalman):
    """A class that extends to non linear models"""

    def __init__(self, emulator, observations, observation_times,
                 observation_metadata, output_array, output_unc, n_params=7):

        LinearKalman.__init__(self, observations, observation_times,
                              observation_metadata, output_array, output_unc,
                              n_params=n_params)
        self.emulator = emulator

    def create_observation_operator(self, metadata, x_forecast):
        """Using an emulator of the nonlinear model around `x_forecast`"""
        n_times = x_forecast.shape[0]/self.n_params
        good_obs = metadata.mask.sum()
        y = np.zeros((good_obs)) # Model evaluated at `x_forecast`
        H_matrix = sp.lil_matrix ( shape=(good_obs, self.n_params*good_obs ),
                                 dtype=np.float32)

        for i in xrange(n_times):
            y0, H0 = self.emulator.predict(x_forecast[i::n_times], do_unc=False)
            y[i] = y0
            dH[i, :] = H0
            ilocs = [i+j*self.n_params for j in xrange(self.n_params)]
            H_matrix[i, (ilocs)] = H[:]
        return H_matrix.tocsr()

    def _get_observations_timestep(self, timestep, band=None):

        if band == 0:
            band = "vis"
        elif band == 1:
            band = "nir"
        time_loc = self.observations.doys == timestep
        fich = self.observations.fnames_a1[time_loc]
        to_BHR = np.array([1.0, 0.189184, -1.377622])
        fname = 'HDF4_EOS:EOS_GRID:"' + \
            '{0:s}":MOD_Grid_BRDF:BRDF_Albedo_Parameters_{1:s}'.format(fich,
                                                                       band)
        g = gdal.Open(fname)
        data = g.ReadAsArray()
        mask = data != 32767
        data[mask] *= 0.001
        bhr = np.where(mask,
                                  data * to_BHR[:, None, None], np.nan)
        fich = self.observations.fnames_a2[time_loc]
        fname = 'HDF4_EOS:EOS_GRID:' + \
                '"{0:s}":MOD_Grid_BRDF:BRDF_Albedo_Quality'.format(fich)
        g = gdal.Open(fname)
        qa = g.ReadAsArray()
        fname = 'HDF4_EOS:EOS_GRID:' + \
                '"{0:s}":MOD_Grid_BRDF:Snow_BRDF_Albedo'.format(fich)
        g = gdal.Open(fname)
        snow = g.ReadAsArray()
        # qa used to define R_mat **and** mask. Don't know what to do with
        # snow information really... Ignore it?
        mask = mask * (qa != 255) # This is OK pixels


        R_mat[qa == 0] = np.max (2.5e-3, bhr * 0.05)
        R_mat[qa == 1] = np.max (2.5e-3, bhr * 0.07)

        metadata = Metadata(mask, R_mat)

        return bhr, R_mat, mask, metadata


if __name__ == "__main__":
    import glob


    files = glob.glob("/storage/ucfajlg/Aurade_MODIS/MCD43A1*.hdf")
    file.sort()
    fnames_a1 = []
    fnames_a2 = []
    doys = []
    for fich in files:
        fname = fich.split ("/")[-1]
        doy = int (fname.split(".")[1][-3:])
        fnames_a1.append (fname)
        fnames_a2.append (fname.replace ("MCD43A1", "MCD43A2"))
        doys.append (doy)
    mcd43_observations= MCD43_observations(doys, fnames_a1, fnames_a2)




