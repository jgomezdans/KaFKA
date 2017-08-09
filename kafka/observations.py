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

import glob
import os
import datetime
from collections import namedtuple

import numpy as np
import gdal
from scipy.ndimage import zoom

from kernels import Kernels
from BRDF_descriptors import RetrieveBRDFDescriptors



__author__ = "J Gomez-Dans"
__copyright__ = "Copyright 2017 J Gomez-Dans"
__version__ = "1.0 (09.03.2017)"
__license__ = "GPLv3"
__email__ = "j.gomez-dans@ucl.ac.uk"


MOD09_data = namedtuple ("MOD09_data", 
                         "reflectance mask uncertainty obs_op sza vza raa")
BHR_data = namedtuple("BHR_data",
                        "albedo mask uncertainty obs_op")

def get_modis_dates (fnames):
    """Extract MODIS dates from filenames"""
    dates = []
    for fname in fnames:
        txt_string = os.path.basename(fname).split(".")[1][1:]
        date = datetime.datetime.strptime(txt_string, "%Y%j")
        dates.append(date)
        
    return dates




class MOD09_Observations(object):
    """A generic M*D09 data reader"""
    def __init__ (self, dates, filenames):
        if not len(dates) == len(filenames):
            raise ValueError("{} dates, {} filenames".format(
                len(dates), len(filenames)))
        self.dates = dates # e.g. a list of datetimes
        self.filenames = filenames # a list of files
    
    def get_band_data(self, the_date, band_no):
        """Returns observations for a given band, uncertainty, mask and 
        observation operator."""
        QA_OK = np.array([8, 72, 136, 200, 1032, 1288, 2056, 2120, 
                          2184, 2248])
        unc = [0.004, 0.015, 0.003, 0.004, 0.013, 0.010, 0.006]
        iloc = self.dates.index(the_date)
        fname = self.filenames[iloc] # Get the HDF filename
        # Read in reflectance
        g = gdal.Open('HDF4_EOS:EOS_GRID:"{}"'.format(fname) + 
                      ':MODIS_Grid_500m_2D:sur_refl_b0{}_1'.format(band_no))
        refl = g.ReadAsArray()/10000. # I think it was 10000...
        # Read in QA MODIS_Grid_1km_2D:state_1km_1
        g = gdal.Open('HDF4_EOS:EOS_GRID:"{}"'.format(fname) + 
                      ':MODIS_Grid_1km_2D:state_1km_1')
        qa = g.ReadAsArray() 
        
        mask = np.in1d(qa, QA_OK).reshape((1200,1200))

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
        raa = vaa - saa # I think...
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
    


class BHRObservations(BRDF_descriptors):
    def __init__ (self, emulator, tile, mcd43a1_dir, start_time, end_time=None, 
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
         super().__init__(tile, mcd43a1_dir, start_time, end_time, 
                          mcd43a2_dir)
         
    def get_band_data(self, the_date, band_no):
        
        to_BHR = np.array([1.0, 0.189184, -1.377622])
        kernels, mask, qa_level = self.get_brdf_descriptors(band_no, date)
        bhr = np.where(mask,
                       kernels * to_BHR[:, None, None], np.nan).sum(axis=0)
        R_mat = np.zeros_like(bhr)
        R_mat[qa_level == 0] = np.maximum(2.5e-3, bhr[qa_level == 0] * 0.05)
        R_mat[qa_level == 1] = np.maximum(2.5e-3, bhr[qa_level == 1] * 0.07)
        R_mat[np.logical_not(mask)] = 0.
        
        bhr_data = BHR_data(bhr, mask, R_mat, None)
        return bhr_data
