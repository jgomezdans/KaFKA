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
from scipy.ndimage import zoom
import kernels
from linear_kf import LinearKalman

# metadata is now different as it has angles innit
Metadata = namedtuple('Metadata', 'mask uncertainty vza sza raa')
MODIS_observations = namedtuple('MODIS_observations',
                        'doys qa vza sza saa vaa b01 b02 b03 b04 b05 b06 b07')



class KernelLinearKalman (LinearKalman):
    """A class that extends to linear kernel models"""

    def __init__(self, observations, observation_times,
                 observation_metadata, output_array, output_unc):

        LinearKalman.__init__(self, observations, observation_times,
                observation_metadata, output_array, output_unc, n_params=3)

    def create_observation_operator(self, metadata, x_forecast):
        """The linear kernel models..."""

        K = kernels.Kernels(metadata.vza[metadata.mask],
                            metadata.sza[metadata.mask],
                            metadata.raa[metadata.mask],
                            LiType="Sparse",
                            doIntegrals=False, normalise=1, RecipFlag=True,
                            RossHS=False, MODISSPARSE=True, RossType="Thick")
        good_obs = metadata.mask.sum() # size of H_matrix
        zz = np.zeros(good_obs)
        data = np.c_[np.r_[np.ones(good_obs), zz, zz],
                            np.r_[zz, K.Ross[:], zz],
                            np.r_[zz, zz, K.Li[:]]].T
        offsets = [ 0, good_obs, 2*good_obs]
        H_matrix = sp.dia_matrix((data, offsets),
                                 shape=(good_obs, self.n_params*good_obs ),
                                 dtype=np.float32)
        # The following arrangement is all isotropics, all volumetrics and then
        # all geometrics. For the non-linear model, it might be better to have
        # all data per grid cell, and not all clobbered together.
        return H_matrix.tocsr()


if __name__ == "__main__":
    import gdal
    import scipy.ndimage
    import glob

    files = glob.glob("/storage/ucfajlg/Aurade_MODIS/TERRA/2010/MOD09*hdf")
    files.sort()
    metadata = []
    obs_time = []
    b2 = []
    n = 1500

    for ii,fname in enumerate(files):
        if ii > 5:
            break
        the_date = int(fname.split("/")[-1].split(".")[1][-3:]) - 209
        obs_time.append(the_date)
        g = gdal.Open(('HDF4_EOS:EOS_GRID:"{0}":' +
                      'MODIS_Grid_500m_2D:sur_refl_b02_1').format(fname))
        b2.append(g.ReadAsArray()[:n,:n]/10000.)
        g = gdal.Open(('HDF4_EOS:EOS_GRID:"{0}":' +
                      ':MODIS_Grid_1km_2D:state_1km_1').format(fname))
        qa = g.ReadAsArray()
        g = gdal.Open(('HDF4_EOS:EOS_GRID:"{0}":' +
                      ':MODIS_Grid_1km_2D:SensorZenith_1').format(fname))
        vza = g.ReadAsArray()/100.
        g = gdal.Open(('HDF4_EOS:EOS_GRID:"{0}":' +
                      ':MODIS_Grid_1km_2D:SensorAzimuth_1').format(fname))
        vaa = g.ReadAsArray()/100.
        g = gdal.Open(('HDF4_EOS:EOS_GRID:"{0}":' +
                      ':MODIS_Grid_1km_2D:SolarZenith_1').format(fname))
        sza = g.ReadAsArray()/100.
        g = gdal.Open(('HDF4_EOS:EOS_GRID:"{0}":' +
                      ':MODIS_Grid_1km_2D:SolarAzimuth_1').format(fname))
        saa = g.ReadAsArray()/100.
        raa = vaa - saa
        QA_OK = np.array( [ 8, 72, 136, 200, 1032, 1288, 2056, 2120, 2184, 2248 ] )
        mask = np.in1d(qa, QA_OK)
        # Resample to 500m
        mask = scipy.ndimage.zoom ( mask.reshape((1200,1200)), 2, order=0)
        vza = scipy.ndimage.zoom(vza, 2, order=0)
        sza = scipy.ndimage.zoom(sza, 2, order=0)
        raa = scipy.ndimage.zoom(raa, 2, order=0)
        # Lump into metadata container
        metadata.append(Metadata(mask[:n,:n], 0.015, vza[:n,:n],
                                 sza[:n,:n], raa[:n,:n]))
        #metadata.mask.append ( mask )
        #metadata.sza.append(sza)
        #metadata.vza.append(vza)
        #metadata.raa.append(raa)

    obs_time = np.array(obs_time)
    #metadata = Metadata(mask, 0.015, vza, sza, raa)
    output_array=np.zeros((20,3, n, n))
    output_unc= np.zeros((20, 3, n, n))
    kalman_filter = KernelLinearKalman(np.array(b2), obs_time, metadata,
                                       output_array,
                                       output_unc)
    kalman_filter.set_trajectory_model()
    Q = np.ones(n*n*3)*.005
    Q[:(n*n)] *= 1.# isotropic has more model noise
    kalman_filter.set_trajectory_uncertainty(Q)

    x_f = 0.5 * np.ones(3*n*n).ravel()
    P_f = sp.eye(3*n*n, 3*n*n, format="csc", dtype=np.float32)

    kalman_filter.run(x_f, P_f, refine_diag=False)
