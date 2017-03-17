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
from nonlinear_ekf import NonLinearKalman
import gp_emulator
import gdal

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# metadata is now different as it has angles innit
Metadata = namedtuple('Metadata', 'mask uncertainty band')
MCD43_observations = namedtuple('MCD43_observations',
                        'doys mcd43a1 mcd43a2')

class BHRKalman (NonLinearKalman):
    """The non-linear EKF working on MODIS MCD43 C5 data"""
    def __init__(self, emulator, observations, observation_times,
                 observation_metadata, output_array, output_unc,
                 bands_per_observation=2, diagnostics=True,n_params=7):

        NonLinearKalman.__init__(self, emulator, observations, observation_times,
                                observation_metadata, output_array, output_unc,
                                bands_per_observation=bands_per_observation,
                                diagnostics=diagnostics, n_params=n_params)

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
        #time_loc = self.observations.doys == timestep
        fich = self.observations.mcd43a1[timestep]
        to_BHR = np.array([1.0, 0.189184, -1.377622])
        fname = 'HDF4_EOS:EOS_GRID:"' + \
                '{0:s}":MOD_Grid_BRDF:BRDF_Albedo_Parameters_{1:s}'.format(fich,
                                                                           band)
        g = gdal.Open(fname)
        data = g.ReadAsArray()[:, :512, :512]
        mask = np.all(data != 32767, axis=0)
        data = np.where(mask, data * 0.001, np.nan)

        bhr = np.where(mask,
                       data * to_BHR[:, None, None], np.nan).sum(axis=0)
        fich = self.observations.mcd43a2[timestep]
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
        mask = mask * (qa != 255)  # This is OK pixels
        R_mat = bhr * 0.0

        R_mat[qa == 0] = np.maximum(2.5e-3, bhr[qa == 0] * 0.05)
        R_mat[qa == 1] = np.maximum(2.5e-3, bhr[qa == 1] * 0.07)
        R_mat_sp = sp.csr_matrix((mask.sum(), mask.sum()))
        R_mat_sp.setdiag(R_mat[mask])
        
        metadata = Metadata(mask, R_mat_sp, band)

        return bhr, R_mat_sp, mask, metadata


    def _set_plot_view(self, diag_str, timestep, observations):
        obj = namedtuple("PlotObject", "fig axs fname nx ny")
        title = "%s %d" % (diag_str, timestep)
        fname = "diagnostic_%s_%04d" % (diag_str, timestep)
        # Only bothering with LAIeff for the time being!
        fig, axs = plt.subplots(ncols=self.n_params, nrows=2, sharex=True,
                                sharey=True, figsize=(15, 5),
                                subplot_kw=dict(
                                    adjustable='box-forced', aspect='equal'))
        #axs = axs.flatten() # Easier *not* to flatten
        fig.suptitle(title)
        ny, nx = observations.shape
        plot_obj = obj(fig, axs, fname, nx, ny)
        return plot_obj


    def _plotter_iteration_start(self, plot_obj, x, obs, mask):
        cmap = plt.cm.viridis
        cmap.set_bad = "0.8"
        
        #plot_obj.axs[0].imshow(x[6*512*512:].reshape(obs.shape), interpolation='nearest',
        #                       cmap=cmap)
        #plot_obj.axs[0].set_title("Prior state")
        
        plot_obj.axs[0][0].imshow(obs, interpolation='nearest',
                               cmap=cmap)
        plot_obj.axs[0][0].set_title("Observations")


    def _plotter_iteration_end(self, plot_obj, x, P, innovation, mask):
        cmap = plt.cm.viridis
        cmap.set_bad = "0.8"

        M = np.ones((plot_obj.ny, plot_obj.nx)) * np.nan
        not_masked = mask.reshape((plot_obj.ny, plot_obj.nx))
        M[not_masked] = innovation
        #plot_obj.axs[2].imshow(M, interpolation='nearest',
                               #cmap=cmap)
        #plot_obj.axs[2].set_title("Innovation")
        n_pixels = plot_obj.nx*plot_obj.ny
        for i in xrange(self.n_params):
            plot_obj.axs[1][i].imshow(x[(i*n_pixels):((i+1)*n_pixels)].reshape
                                      ((plot_obj.ny, plot_obj.nx)),
                               interpolation='nearest', cmap=cmap)
        #plot_obj.axs[3].set_title("Posterior mean")
        #unc = P.diagonal().reshape((plot_obj.ny, plot_obj.nx))
        #plot_obj.axs[4].imshow(np.sqrt(unc),
        #                       interpolation='nearest', cmap=cmap)
        #plot_obj.axs[4].set_title("Posterior StdDev")

        plot_obj.fig.savefig(plot_obj.fname + ".png", dpi=72,
                             bbox_inches="tight")
        plt.close(plot_obj.fig)


if __name__ == "__main__":
    import glob
    import cPickle

    files = glob.glob("/storage/ucfajlg/Aurade_MODIS/MCD43/MCD43A1.A2010*.hdf")
    files.sort()
    fnames_a1 = []
    fnames_a2 = []
    doys = []
    for fich in files:
        fname = fich.split("/")[-1]
        doy = int(fname.split(".")[1][-3:])
        fnames_a1.append(fich)
        fnames_a2.append(fich.replace("MCD43A1", "MCD43A2"))
        doys.append(doy)
    doys = np.array(doys)
    mcd43_observations = MCD43_observations(doys, fnames_a1, fnames_a2)
    emulator = cPickle.load(open(
        "../SAIL_emulator_both_500trainingsamples.pkl", 'r'))
    kalman = BHRKalman(emulator, mcd43_observations, doys,
                             mcd43_observations, [], [], n_params=7)

    # test methods
    bhr, R_mat, mask, metadata = kalman._get_observations_timestep(1,
                                                                   band=0)
    x0 = np.ones(7 * 512 * 512) * 0.5
    P_forecast = sp.dia_matrix((7*512*512, 7*512*512))
    P_forecast.setdiag(np.ones(7 * 512 * 512) * 0.1)
    kalman.run(x0, P_forecast,
                   diag_str="diagnostics",
                   approx_diagonal=True, refine_diag=False,
                   iter_obs_op=False, is_robust=False)

