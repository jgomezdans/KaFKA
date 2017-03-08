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
import gp_emulator
import gdal
# metadata is now different as it has angles innit
Metadata = namedtuple('Metadata', 'mask uncertainty band')
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
        """Using an emulator of the nonlinear model around `x_forecast`.
        This case is quite special, as I'm focusing on a BHR SAIL 
        version (or the JRC TIP), which have spectral parameters 
        (e.g. leaf single scattering albedo in two bands, etc.). This
        is achieved by using the `state_mapper` to select which bits
        of the state vector (and model Jacobian) are used."""
        
        n_times = x_forecast.shape[0]/self.n_params
        good_obs = metadata.mask.sum()
        
        
        H_matrix = sp.lil_matrix ( (good_obs, self.n_params*good_obs),
                                 dtype=np.float32)
        # So the model has spectral components. 
        if metadata.band == "vis":
            # ssa, asym, LAI, rsoil
            state_mapper = np.array([0,1,6,2])
        elif metadata.band == "nir":
            # ssa, asym, LAI, rsoil
            state_mapper = np.array([3,4,6,5])
        
        x0 = np.zeros((good_obs, 4))
        for i,j in enumerate(state_mapper):
            x0[:, i] = (x_forecast[(j*n_times):((j+1)*n_times)]
                        [metadata.mask.ravel()])
        
        _, H0 = self.emulator.predict(x0, do_unc=False)
        for i in xrange(good_obs):
            ilocs = [(i+j*good_obs) for j in state_mapper]
            H_matrix[i, ilocs] = H0[i]
        return H_matrix.tocsr()

    def _get_observations_timestep(self, timestep, band=None):
        """This method is based on the MCD43 family of products.
        It opens the data file, reads in the VIS (... NIR) kernels,
        integrates them to BHR, and also extracts QA flags and the
        snow flag. The QA flags are then converted to uncertainty
        as per Pinty's 5 and 7% argument.
        
        TODO Needs a clearer interface to subset parts of the image,
        as it's currently done rather crudely."""

        if band == 0:
            band = "vis"
        elif band == 1:
            band = "nir"
        time_loc = self.observations.doys == timestep
        fich = self.observations.mcd43a1[time_loc]
        to_BHR = np.array([1.0, 0.189184, -1.377622])
        fname = 'HDF4_EOS:EOS_GRID:"' + \
            '{0:s}":MOD_Grid_BRDF:BRDF_Albedo_Parameters_{1:s}'.format(fich,
                                                                       band)
        g = gdal.Open(fname)
        data = g.ReadAsArray()[:, :512, :512]
        mask = np.all(data != 32767, axis=0)
        data = np.where(mask, data*0.001, np.nan)

        bhr = np.where(mask,
                            data * to_BHR[:, None, None], np.nan).sum(axis=0)
        fich = self.observations.mcd43a2[time_loc]
        fname = 'HDF4_EOS:EOS_GRID:' + \
                '"{0:s}":MOD_Grid_BRDF:BRDF_Albedo_Quality'.format(fich)
        g = gdal.Open(fname)
        qa = g.ReadAsArray()[:512, :512]
        fname = 'HDF4_EOS:EOS_GRID:' + \
                '"{0:s}":MOD_Grid_BRDF:Snow_BRDF_Albedo'.format(fich)
        g = gdal.Open(fname)
        snow = g.ReadAsArray()[:512, :512]
        # qa used to define R_mat **and** mask. Don't know what to do with
        # snow information really... Ignore it?
        mask = mask * (qa != 255) # This is OK pixels
        R_mat = bhr*0.0

        R_mat[qa == 0] = np.maximum (2.5e-3, bhr[qa==0] * 0.05)
        R_mat[qa == 1] = np.maximum (2.5e-3, bhr[qa==1] * 0.07)

        metadata = Metadata(mask, R_mat, band)

        return bhr, R_mat, mask, metadata


if __name__ == "__main__":
    import glob
    import cPickle
    

    files = glob.glob("/storage/ucfajlg/Aurade_MODIS/MCD43/MCD43A1.A2010*.hdf")
    files.sort()
    fnames_a1 = []
    fnames_a2 = []
    doys = []
    for fich in files:
        fname = fich.split ("/")[-1]
        doy = int (fname.split(".")[1][-3:])
        fnames_a1.append (fich)
        fnames_a2.append (fich.replace ("MCD43A1", "MCD43A2"))
        doys.append (doy)
    mcd43_observations= MCD43_observations(doys, fnames_a1, fnames_a2)
    emulator = cPickle.load (open(
        "../SAIL_emulator_both_500trainingsamples.pkl", 'r'))
    kalman = NonLinearKalman(emulator, mcd43_observations, doys,
                 mcd43_observations, [], [], n_params=7)

    # test methods
    bhr, R_mat, mask, metadata = kalman._get_observations_timestep(1, 
                                                                   band=0)
    x0 = np.ones(7*512*512)*0.5
    H=kalman.create_observation_operator(metadata, x0)
    
