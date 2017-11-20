#!/usr/bin/env python
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

"""
We need an observations class that returns the observations, the mask,
the uncertainty, the observation operator, other relevant metadata (angles)
and maybe even a spatial factor (not for Synergy but for multiply).

A lightweight class (e.g. namedtuple) might be enough for this, plus
a bunch of functions (or a class) that are specialised in different
observational streams.

If implemented as a class, it might be possible to have a method
to "relinearise" the model around a different point... However, this
starts calling for state grids and what not, and is probably best done
in the (E)KF class.

In reality, they should be indexed by time, so we will have observationS
plural. Start by defining

M*D09GA
MCD43A1/2 -> See BRDF_descriptors!

"""

import cPickle
import datetime
import glob
import os
from collections import namedtuple

import gdal

import numpy as np

from BRDF_descriptors import RetrieveBRDFDescriptors

from kernels import Kernels

import scipy.sparse as sp
from scipy.ndimage import zoom

os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'

__author__ = "J Gomez-Dans"
__copyright__ = "Copyright 2017 J Gomez-Dans"
__version__ = "1.0 (09.03.2017)"
__license__ = "GPLv3"
__email__ = "j.gomez-dans@ucl.ac.uk"

MOD09_data = namedtuple("MOD09_data",
                        "reflectance mask uncertainty obs_op sza vza raa")
BHR_data = namedtuple("BHR_data",
                      "observations mask uncertainty metadata emulator")


def get_modis_dates(fnames):
    """Extract MODIS dates from filenames"""
    dates = []
    for fname in fnames:
        txt_string = os.path.basename(fname).split(".")[1][1:]
        date = datetime.datetime.strptime(txt_string, "%Y%j")
        dates.append(date)

    return dates

# TODO needs class for MODIS L1b product too
# These classes should define emulators


class MOD09_ObservationsKernels(object):
    """A generic M*D09 data reader"""
    def __init__(self, dates, filenames):
        if not len(dates) == len(filenames):
            raise ValueError("{} dates, {} filenames".format(
                len(dates), len(filenames)))
        self.dates = dates  # e.g. a list of datetimes
        self.filenames = filenames  # a list of files

    def get_band_data(self, the_date, band_no):
        """Returns observations for a given band, uncertainty, mask and
        observation operator."""
        QA_OK = np.array([8, 72, 136, 200, 1032, 1288, 2056, 2120,
                          2184, 2248])
        unc = [0.004, 0.015, 0.003, 0.004, 0.013, 0.010, 0.006]
        try:
            iloc = self.dates.index(the_date)
        except ValueError:
            # No observations found
            return None
        fname = self.filenames[iloc]  # Get the HDF filename
        # Read in reflectance
        g = gdal.Open('HDF4_EOS:EOS_GRID:"{}"'.format(fname) +
                      ':MODIS_Grid_500m_2D:sur_refl_b0{}_1'.format(band_no))
        refl = g.ReadAsArray()/10000.  # I think it was 10000...
        # Read in QA MODIS_Grid_1km_2D:state_1km_1
        g = gdal.Open('HDF4_EOS:EOS_GRID:"{}"'.format(fname) +
                      ':MODIS_Grid_1km_2D:state_1km_1')
        qa = g.ReadAsArray()

        mask = np.in1d(qa, QA_OK).reshape((1200, 1200))

        # TODO Need to convert QA to True/False mask
        # Read in angles
        g = gdal.Open('HDF4_EOS:EOS_GRID:"{}"'.format(fname) +
                      ':MODIS_Grid_1km_2D:SolarZenith_1')
        sza = g.ReadAsArray()/100.
        g = gdal.Open('HDF4_EOS:EOS_GRID:"{}"'.format(fname) +
                      ':MODIS_Grid_1km_2D:SolarAzimuth_1')
        saa = g.ReadAsArray()/100.
        g = gdal.Open('HDF4_EOS:EOS_GRID:"{}"'.format(fname) +
                      ':MODIS_Grid_1km_2D:SensorZenith_1')
        vza = g.ReadAsArray()/100.
        g = gdal.Open('HDF4_EOS:EOS_GRID:"{}"'.format(fname) +
                      ':MODIS_Grid_1km_2D:SensorAzimuth_1')
        vaa = g.ReadAsArray()/100.
        raa = vaa - saa  # I think...
        # Needs a zoom to make it 2400*2400
        raa = zoom(raa, 2, order=0)
        vza = zoom(vza, 2, order=0)
        sza = zoom(sza, 2, order=0)
        mask = zoom(mask, 2, order=0)
        K = Kernels(vza, sza, raa, LiType="Sparse", doIntegrals=False,
                    normalise=1, RecipFlag=True,
                    RossHS=False, MODISSPARSE=True, RossType="Thick")
        uncertainty = refl*0 + unc[band_no-1]
        data_object = MOD09_data(refl, mask, uncertainty, K, sza, vza, raa)

        return data_object


class SynergyKernels(object):
    """An object to store, process and update linear kernel weights datasets
    produced by the Synergy processing chain"""
    def __init__(self, directory, tile, start_time, end_time=None):

        fnames = glob.glob("%s/*.%s*_b0_kernel_weights.tif" %
                           (directory, tile))
        dates = []
        kernels = []
        uncertainties = []
        masks = []
        for fname in fnames:
            txt_string = os.path.basename(fname).split(".")[1][1:]
            date = datetime.datetime.strptime(txt_string, "%Y%j")
            if (start_time >= date) and ((end_time is None) or
                                         (date <= end_time)):
                dates.append(date)
                kernels.append(fname)
                uncertainties.append(fname.replace("kernel_weights",
                                                   "kernel_unc"))
                masks.append(fname.replace("_b0_kernel_weights", "mask"))
        self.dates = dates
        self.kernels = kernels
        self.uncertainties = uncertainties
        self.masks = masks

    def add_observations(self, the_date, the_kernels, the_uncs, the_mask):
        """Adds observations to the list. Assume the date is datetime object,
        and all files are strings to files that exist."""
        self.dates.append(the_date)
        self.kernels.append(the_kernels)
        self.uncertainties.append(the_uncs)
        self.masks.append(the_mask)

    def get_band_data(self, the_date, band_no):
        """Assume `band_no` is 0 for VIS and 1 for NIR (BB)"""
        # the integrals of the kernels
        to_BHR = np.array([1.0, 0.189184, -1.377622])
        # the spectral integration for BB (MODIS bands)
        to_VIS = np.array([0.3265, 0., 0.4364, 0.2366, 0, 0, 0])
        a_to_VIS = -0.0019
        to_NIR = np.array([0., 0.5447, 0, 0, 0.1363, 0.0469, 0.2536])
        a_to_NIR = -0.0068

        # find the requested date
        date_idx = self.dates.index(the_date)
        BHR = []
        for band in xrange(7):
            g = gdal.Open(self.kernels[date_idx].replace(
                "b0", "b%d" % band))
            kernels = g.ReadAsArray()  # 3*nx*ny
            BHR.append(np.sum(kernels * to_BHR[:, None, None], axis=0))
            # Taking the mask into account, we can add a where statement
            # np.where(mask,
            #          kernels * to_BHR[:, None, None], np.nan).sum(axis=0)
            # Uncertainty is also straightforward if no correlation is assumed
        BHR = np.array(BHR)
        # Under the assumption that kernels are
        if band_no == 0:  # VIS
            BHR = np.sum(BHR*to_VIS, axis=0) + a_to_VIS
        elif band_no == 1:
            BHR = np.sum(BHR * to_NIR, axis=0) + a_to_NIR


class BHRObservations(RetrieveBRDFDescriptors):
    def __init__(self, emulator, tile, mcd43a1_dir,
                 start_time, ulx=0, uly=0, dx=2400, dy=2400, end_time=None,
                 mcd43a2_dir=None):
        """The class needs to locate the data granules. We assume that
        these are available somewhere in the filesystem and that we can
        index them by location (MODIS tile name e.g. "h19v10") and
        time. The user can give a folder for the MCD43A1 and A2 granules,
        and if the second is ignored, it will be assumed that they are
        in the same folder. We also need a starting date (either a
        datetime object, or a string in "%Y-%m-%d" or "%Y%j" format. If
        the end time is not specified, it will be set to the date of the
        latest granule found."""

        # Call the constructor first
        # Python2
        RetrieveBRDFDescriptors.__init__(self, tile,
                                         mcd43a1_dir, start_time, end_time,
                                         mcd43a2_dir)
        # Python3
        #  super().__init__(tile, mcd43a1_dir, start_time, end_time,
        #                  mcd43a2_dir)
        self._get_emulator(emulator)
        self.dates = sorted(self.a1_granules.keys())
        self.dates = self.dates[::8]
        a1_temp = {}
        a2_temp = {}
        for k in self.dates:
            a1_temp[k] = self.a1_granules[k]
            a2_temp[k] = self.a2_granules[k]
        self.a1_granules = a1_temp
        self.a2_granules = a2_temp
        self.band_transfer = {0: "vis",
                              1: "nir"}
        self.ulx = ulx
        self.uly = uly
        self.dx = dx
        self.dy = dy

    def define_output(self):
        reference_fname = self.a1_granules[self.dates[0]]
        g = gdal.Open('HDF4_EOS:EOS_GRID:' +
                      '"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_vis' %
                      reference_fname)
        proj = g.GetProjection()
        geoT = np.array(g.GetGeoTransform())
        new_geoT = geoT*1.
        new_geoT[0] = new_geoT[0] + self.ulx*new_geoT[1]
        new_geoT[3] = new_geoT[3] + self.uly*new_geoT[5]
        return proj, new_geoT.tolist()

    def _get_emulator(self, emulator):
        if not os.path.exists(emulator):
            raise IOError("The emulator {} doesn't exist!".format(emulator))
        # Assuming emulator is in an pickle file...
        self.emulator = cPickle.load(open(emulator, 'rb'))

    def get_band_data(self, the_date, band_no):

        to_BHR = np.array([1.0, 0.189184, -1.377622])
        retval = self.get_brdf_descriptors(band_no, the_date)
        if retval is None:  # No data on this date
            return None
        kernels, mask, qa_level = retval
        mask = mask[self.uly:(self.uly + self.dy),
                    self.ulx:(self.ulx + self.dx)]
        qa_level = qa_level[self.uly:(self.uly + self.dy),
                            self.ulx:(self.ulx + self.dx)]
        kernels = kernels[:, self.uly:(self.uly + self.dy),
                          self.ulx:(self.ulx + self.dx)]
        bhr = np.where(mask,
                       kernels * to_BHR[:, None, None], np.nan).sum(axis=0)
        R_mat = np.zeros_like(bhr)

        R_mat[qa_level == 0] = np.maximum(2.5e-3, bhr[qa_level == 0] * 0.05)
        R_mat[qa_level == 1] = np.maximum(2.5e-3, bhr[qa_level == 1] * 0.07)
        R_mat[np.logical_not(mask)] = 0.
        N = mask.ravel().shape[0]
        R_mat_sp = sp.lil_matrix((N, N))
        R_mat_sp.setdiag(1./(R_mat.ravel())**2)
        R_mat_sp = R_mat_sp.tocsr()

        bhr_data = BHR_data(bhr, mask, R_mat_sp, None, self.emulator)
        return bhr_data


class KafkaOutput(object):
    """A very simple class to output the state."""
    def __init__(self, tilewidth, geotransform, projection, folder,
                 fmt="GTiff"):
        """The inference engine works on tiles, so we get the tilewidth
        (we assume the tiles are square), the GDAL-friendly geotransform
        and projection, as well as the destination directory and the
        format (as a string that GDAL can understand)."""
        self.tilewidth = tilewidth
        self.geotransform = geotransform
        self.projection = projection
        self.folder = folder
        self.fmt = fmt

    def dump_data(self, timestep, x_analysis, P_analysis, P_analysis_inv):
        drv = gdal.GetDriverByName(self.fmt)
        for ii, param in enumerate(["w_vis", "x_vis", "a_vis",
                                    "w_nir", "x_nir", "a_nir", "TeLAI"]):
            fname = os.path.join(self.folder, "%s_%s.tif" %
                                 (param, timestep.strftime("A%Y%j")))
            dst_ds = drv.Create(fname, self.tilewidth, self.tilewidth, 1,
                                gdal.GDT_Float32, ['COMPRESS=DEFLATE',
                                                   'BIGTIFF=YES',
                                                   'PREDICTOR=1', 'TILED=YES'])
            dst_ds.SetProjection(self.projection)
            dst_ds.SetGeoTransform(self.geotransform)
            dst_ds.GetRasterBand(1).WriteArray(x_analysis[ii::7].reshape((
                self.tilewidth, self.tilewidth)))


if __name__ == "__main__":
    emulator = "../SAIL_emulator_both_500trainingsamples.pkl"
    tile = "h17v05"
    start_time = "2017001"
    mcd43a1_dir = "/storage/ucfajlg/Ujia/MCD43/"
    bhr_data = BHRObservations(emulator, tile, mcd43a1_dir, start_time,
                               end_time=None, mcd43a2_dir=None)
    vis = bhr_data.get_band_data(datetime.datetime(2017, 8, 1), "vis")
    nir = bhr_data.get_band_data(datetime.datetime(2017, 8, 1), "nir")
